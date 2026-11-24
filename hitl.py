"""Завдання 4. Human-in-the-Loop: interrupt() перед ризиковим tool book_hotel.

Потік: executor → LLM просить book_hotel → вузол approval → interrupt() зупиняє граф (стан — у SqliteSaver)
→ людина відповідає Command(resume={...}) → approve: tools виконує бронювання; edit: аргументи змінюються, і
approval питає ще раз уже з новою сумою; reject: tool НЕ виконується, executor отримує ToolMessage з причиною.
Людину не питають, якщо бронювання однаково неможливе (невалідні аргументи, немає вільних номерів).
"""

import json
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from tools import BookHotelInput, booking_summary, has_rooms

RISKY_TOOLS = {"book_hotel"}  # незворотні дії: лише після підтвердження людини


class HumanDecision(BaseModel):
    """Відповідь людини — значення Command(resume=...). Валідуємо: це вхід ззовні графа."""

    action: Literal["approve", "reject", "edit"]
    reason: str = Field(default="", description="Причина відмови (для reject)")
    args: dict = Field(default_factory=dict,
                       description='Аргументи, які треба змінити (для edit), напр. {"breakfast": true}')


def describe_booking(b: BookHotelInput) -> str:
    """Що саме буде заброньовано і за скільки — людина має бачити суму, а не лише сирі аргументи.
    Той самий формат, що й у підтвердженні book_hotel."""
    return booking_summary(b.hotel_id, b.check_in, b.check_out, b.guests, b.breakfast, b.guest_name, b.email)


def approval_node(state) -> Command[Literal["tools", "executor", "approval"]]:
    """Зупиняє граф перед ризиковою дією, показує деталі й чекає рішення людини.

    При resume LangGraph перезапускає вузол з початку, а interrupt() повертає відповідь людини. Тому interrupt
    стоїть в окремому вузлі без викликів LLM: інакше LLM викликалась би вдруге і могла б повернути інші аргументи,
    ніж ті, що людина підтвердила.
    """
    ai = state["step_messages"][-1]
    call = ai.tool_calls[0]  # executor вимикає паралельні виклики, тож tool call рівно один
    try:
        booking = BookHotelInput.model_validate(call["args"])
    except ValidationError:
        # Невалідні аргументи: людину не турбуємо — ToolNode поверне помилку валідації LLM, і та виправить виклик
        return Command(goto="tools")
    if not has_rooms(booking.hotel_id, booking.guests):
        # Підтверджувати нічого: book_hotel однаково відмовить («Бронювання неможливе») і LLM запропонує інший готель
        return Command(goto="tools")

    decision = HumanDecision.model_validate(interrupt({
        "tool": call["name"],
        "args": booking.model_dump(mode="json"),  # нормалізовані: дати YYYY-MM-DD, усі поля з дефолтами
        "summary": describe_booking(booking),
        "question": "approve — виконати; reject — відхилити (з причиною); edit — змінити аргументи "
                    "(нову суму буде показано на підтвердження)",
    }))

    if decision.action == "approve":
        return Command(goto="tools")
    if decision.action == "edit":
        # Той самий id повідомлення → add_messages замінює tool call. Виконувати одразу не можна: зміна (гості,
        # дати, сніданок) змінює суму й кількість номерів, тож approval питає людину ще раз уже з новими деталями.
        # content очищаємо: там блок function_call зі СТАРИМИ аргументами, і langchain-openai (Responses API)
        # надіслав би LLM саме його, а не змінений tool_calls
        edited = ai.model_copy(update={"content": "",
                                       "tool_calls": [{**call, "args": {**call["args"], **decision.args}}]})
        return Command(goto="approval", update={"step_messages": [edited]})
    rejected = ToolMessage(
        f"Оператор ВІДХИЛИВ {call['name']}: {decision.reason or 'причину не вказано'}. "
        "Дію НЕ виконано, не повторюй її.",
        tool_call_id=call["id"], name=call["name"], status="error",
    )
    return Command(goto="executor", update={"step_messages": [rejected]})


def ask_human(payload: dict) -> dict:
    """CLI-підтвердження. Виконати можна лише явним approve/edit — будь-яка інша відповідь означає відмову."""
    print(f"\n⚠️  Потрібне підтвердження: {payload['tool']}\n   {payload['summary']}")
    answer = input("approve / reject / edit? ").strip().lower()
    if answer in ("a", "approve"):
        return {"action": "approve"}
    while answer in ("e", "edit"):  # невалідний JSON — питаємо ще раз, а не падаємо
        try:
            args = json.loads(input('Змінені аргументи JSON, напр. {"breakfast": true}: '))
        except json.JSONDecodeError as e:
            print(f"   Не JSON ({e.msg}), спробуйте ще раз")
            continue
        if isinstance(args, dict):
            return {"action": "edit", "args": args}
        print("   Потрібен JSON-об'єкт з аргументами, напр. {\"guests\": 3}")
    return {"action": "reject", "reason": input("Причина відмови: ").strip()}
