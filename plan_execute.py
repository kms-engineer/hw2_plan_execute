"""Завдання 1–2. Plan-and-Execute агент бронювання готелів у LangGraph зі збереженням стану в SqliteSaver.

Граф:  START → planner → executor ─┬─ tools ──────────────→ executor   (безпечний tool)
                                   ├─ approval (interrupt) → tools | executor | approval   (book_hotel, hitl.py)
                                   └─ replanner ─→ executor | END    (крок виконано)
- planner   — with_structured_output(Plan): ціль і список кроків;
- executor  — виконує ОДИН крок плану: LLM з tools; поки LLM викликає tools — міні-цикл executor ⇄ tools,
              текстова відповідь без tool call = крок виконано;
- tools     — ToolNode; помилки валідації Pydantic повертаються LLM як ToolMessage, граф не падає;
- replanner — with_structured_output(ReplanDecision): continue / replan (замінити кроки, що залишились) / finish.

Запуск:  python plan_execute.py --thread t1 "Знайди готель у Львові з 10 по 12 березня 2027 на двох"
         python plan_execute.py --thread t1        # продовжити тред з останнього checkpoint
"""

import argparse
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from formatting import normalize
from hitl import RISKY_TOOLS, approval_node, ask_human
from knowledge import TOPICS, search_knowledge
from tools import TOOLS as HOTEL_TOOLS

load_dotenv(find_dotenv(usecwd=True))  # OPENAI_API_KEY з .env (шукає вгору від поточної директорії)
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
DB_PATH = Path(__file__).parent / "agent_state.db"
TOOLS = [*HOTEL_TOOLS, search_knowledge]
MAX_PLAN_STEPS = 5    # кроків у плані planner
MAX_REPLAN_STEPS = 4  # нових кроків при replan
MAX_STEPS = 8         # ліміт виконаних кроків (разом з доданими при replan) — захист від нескінченного перепланування
RECURSION_LIMIT = 80  # запобіжник LangGraph: кожен крок = executor + tools + … + replanner (≈ 4–6 переходів)


# ── Structured outputs ───────────────────────────────────────────
class Plan(BaseModel):
    """План виконання запиту користувача."""

    goal: str = Field(description="Головна ціль запиту одним реченням")
    steps: list[str] = Field(description=f"1–{MAX_PLAN_STEPS} послідовних кроків. Кожен — одна дія у форматі "
                                         "«інструмент: що зробити з усіма відомими параметрами»")


class ReplanDecision(BaseModel):
    """Рішення replanner після виконаного кроку."""

    reasoning: str = Field(description="Коротке пояснення рішення на основі результатів кроків")
    action: Literal["continue", "replan", "finish"] = Field(
        description="continue — виконати наступний крок плану як є; replan — замінити кроки, що залишились; "
                    "finish — ціль досягнута або продовжувати немає сенсу")
    updated_steps: list[str] | None = Field(
        default=None, description=f"1–{MAX_REPLAN_STEPS} нових кроків замість тих, що залишились (лише для replan)")
    final_answer: str | None = Field(default=None, description="Фінальна відповідь користувачу (лише для finish)")


# ── State ────────────────────────────────────────────────────────
class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]       # діалог треду: запити + фінальні відповіді (пам'ять між запусками)
    goal: str
    plan: list[str]                               # усі кроки: виконані + заплановані
    current_step: int                             # індекс наступного кроку (0-based) = кількість виконаних
    results: list[str]                            # результати виконаних кроків
    step_messages: Annotated[list, add_messages]  # робочий контекст поточного кроку: tool calls і їх результати
    decision: dict | None                         # останнє рішення replanner (видно в get_state)
    completed: bool


# ── Промпти (статичні: дата й прогрес ідуть окремим повідомленням, щоб не скидати кеш префікса OpenAI) ──
# Єдиний формат даних у планах, результатах кроків і відповідях — такий самий, як у виходах tools
FORMAT_RULES = """Формат даних (однаковий у планах, висновках і відповідях — як у результатах інструментів):
- дати — YYYY-MM-DD, період — «2027-03-10 – 2027-03-12» (не «з … по …» і не «2027-03-10–2027-03-12»);
- суми — ціле число без пробілів з «грн»: «7120 грн»; рейтинг — з крапкою: «9.1»;
- готель — «ID Назва»: «LVI-001 Rynok Square Hotel»; міста — Київ, Львів, Одеса;
- інструменти — точними назвами: search_hotels, get_hotel_details, calculate_total_price, book_hotel,
  search_knowledge;
- простий текст без Markdown (**, #): виводиться в термінал; переліки — рядки з «- »."""

PLANNER_PROMPT = f"""Ти — планувальник агента бронювання готелів в Україні (Київ, Львів, Одеса).
Склади план для ОСТАННЬОГО запиту користувача. Попередні повідомлення — контекст розмови: «він», «цей готель»
означають готель з попередніх відповідей.

Інструменти виконавця:
- search_hotels — пошук готелів з вільними номерами: місто, дати, гості, ціна, рейтинг;
- get_hotel_details — факти про конкретний готель: чи є сніданок і його ціна, чи дозволені тварини, тариф
  скасування, адреса, зручності й теми, де в готелю свої правила;
- calculate_total_price — вартість: номери + сніданки + туристичний збір (без доплат за тварин і депозиту);
- book_hotel — бронювання (потрібні ім'я, прізвище та email гостя). Підтвердження людини система запитує
  автоматично перед виконанням — окремих кроків для підтвердження не додавай;
- search_knowledge — правила проживання за темами ({', '.join(TOPICS)}): без ID готелю — стандартні
  правила мережі, з ID — ще й правила цього готелю, які замінюють загальні.

Правила:
1. 1–{MAX_PLAN_STEPS} кроків. Кожен крок — одна дія у форматі «інструмент: що зробити» з усіма відомими параметрами
   (місто, дати YYYY-MM-DD, гості, ID готелю). Не додавай кроків, яких запит не потребує.
2. Дані конкретних готелів, ціни й наявність — лише search_hotels / get_hotel_details / calculate_total_price.
   Правила — search_knowledge: питання без конкретного готелю («о котрій заселення», «які документи»)
   стосується стандартних правил мережі — НЕ проси уточнити готель; правила конкретного готелю — з його ID.
3. Якщо крок залежить від попереднього — посилайся на нього: «calculate_total_price: для готелю з кроку 1».
4. book_hotel — лише якщо користувач явно просить забронювати і дав ім'я та email.
5. Якщо запит поза доменом або незрозуміло, про що йдеться — один крок «відповісти без інструментів: …».
   Скасувати чи змінити бронювання агент не може: план — search_knowledge з правилами скасування.
6. Плануй основний шлях і не додавай перевірок, результат яких покаже наступний крок: calculate_total_price сам
   повідомить, що сніданку в готелі немає, book_hotel — що немає номерів. Якщо результат кроку виявить проблему,
   replanner замінить решту плану. Перевірку додавай, лише коли інструмент не вміє відфільтрувати потрібне
   (search_hotels не фільтрує тварин, тож для «з собакою» потрібен get_hotel_details).

""" + FORMAT_RULES

EXECUTOR_PROMPT = """Ти — виконавець кроків плану агента бронювання готелів в Україні (Київ, Львів, Одеса).
Виконуй ЛИШЕ поточний крок; наступні кроки виконають пізніше.

Правила:
1. Дані про готелі, ціни, наявність і правила бери ТІЛЬКИ з інструментів, нічого не вигадуй.
2. Дані конкретних готелів і ціни — search_hotels / get_hotel_details / calculate_total_price. Правила —
   search_knowledge: передай topics — теми, яких стосується крок (напр. ["заселення", "документи"]); для правил
   конкретного готелю — ще й hotel_id. Без hotel_id відповідають стандартні правила мережі.
   Якщо потрібні дані вже є в результатах попередніх кроків — не викликай tool знову.
3. Помилка валідації — виправ аргументи й повтори, якщо це можливо; інакше поясни, чому крок не виконано.
4. Якщо крок вимагає бронювання — викликай book_hotel. Система сама зупинить виконання й запитає підтвердження
   людини, тож НЕ проси підтвердження текстом. Якщо оператор відхилив дію — НЕ повторюй виклик, повідом про
   відмову та її причину.
5. Коли крок виконано — відповідай текстом без виклику інструментів: висновок кроку в 1–3 реченнях з ключовими
   даними (ID, суми, коди бронювання). Повні результати інструментів додаються до звіту автоматично.
   Не обчислюй нових сум самостійно.

""" + FORMAT_RULES

REPLANNER_PROMPT = f"""Ти — replanner агента бронювання готелів. Після кожного виконаного кроку оціни прогрес:
- continue — наступний крок плану досі актуальний;
- replan — результати змінили ситуацію (немає номерів, готель не підходить, помилка), тож кроки, що залишились,
  треба замінити: дай updated_steps (1–{MAX_REPLAN_STEPS} кроків у форматі «інструмент: що зробити»);
- finish — ціль досягнута або продовжувати немає сенсу (запит неможливо виконати, оператор відхилив дію).

Правила:
1. Якщо кроків не залишилось — finish (або replan, якщо ціль ще не досягнута).
2. Не повторюй кроки, які вже дали результат. Якщо оператор відхилив бронювання — не бронюй знову: finish.
3. Для finish дай final_answer українською, стисло, лише з даних у результатах кроків (ID і назви готелів, дати,
   суми, коди бронювання, правила). Не обчислюй нових сум. Якщо щось не вдалося — чесно скажи і запропонуй
   альтернативу.
4. Перед finish перевір, що виконано ВСІ умови запиту (сніданок, тварини, ціна, рейтинг, дати). Planner планує
   основний шлях, тож невідповідність умові — сигнал до replan: якщо умову можна виконати іншим варіантом
   (інший готель із результатів пошуку), зроби це сам через replan, а не пропонуй «можу пошукати». Виняток —
   бронювання: інший готель бронюй лише тоді, коли користувач сам про це просив.

""" + FORMAT_RULES


def make_llm() -> ChatOpenAI:
    # temperature не задаємо: gpt-6-luna підтримує лише значення за замовчуванням
    return ChatOpenAI(model=MODEL, timeout=60, max_retries=2)


def progress(state: PlanExecuteState) -> str:
    """Контекст для executor і replanner: запит, ціль, план з позначками виконання, результати."""
    k, plan = state["current_step"], state["plan"]
    steps = "\n".join(f"{i + 1}. {s}{' ✓' if i < k else ''}" for i, s in enumerate(plan))
    query = next(m.text for m in reversed(state["messages"]) if m.type == "human")
    return (f"Сьогодні {date.today()}.\nЗапит користувача: {query}\nЦіль: {state['goal']}\n"
            f"План (✓ — виконано, залишилось кроків: {len(plan) - k}):\n{steps}\n"
            f"Результати виконаних кроків:\n" + ("\n".join(state["results"]) or "немає"))


def build_graph(llm=None) -> StateGraph:
    """Зібрати (не скомпільований) граф planner → executor ⇄ tools / approval → replanner."""
    llm = llm or make_llm()
    planner_llm = llm.with_structured_output(Plan)
    replanner_llm = llm.with_structured_output(ReplanDecision)
    # Один tool call за відповідь: HITL підтверджує рівно одну дію, а кожен виклик видно окремо в історії
    executor_llm = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def planner(state: PlanExecuteState) -> dict:
        messages = [SystemMessage(PLANNER_PROMPT), *state["messages"],
                    HumanMessage(f"Сьогодні {date.today()}. Склади план для останнього запиту.")]
        # cast: with_structured_output типізований як dict | модель, а з Pydantic-схемою повертає саме модель
        plan = cast(Plan, planner_llm.invoke(messages))
        # Порожній план (буває на запитах поза доменом) → один крок відповіді, інакше executor не мав би що виконати
        goal = normalize(plan.goal)  # формат даних гарантує код, а не лише промпт (formatting.py)
        steps = [normalize(s) for s in plan.steps[:MAX_PLAN_STEPS]] or [f"відповісти без інструментів: {goal}"]
        # Новий запит у тому самому треді починає новий план; діалог (messages) зберігається
        return {"goal": goal, "plan": steps, "current_step": 0, "results": [],
                "step_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)], "decision": None, "completed": False}

    def executor(state: PlanExecuteState) -> dict:
        k = state["current_step"]
        task = HumanMessage(f"{progress(state)}\n\nВиконай поточний крок {k + 1}: {state['plan'][k]}")
        response = executor_llm.invoke([SystemMessage(EXECUTOR_PROMPT), task, *state["step_messages"]])
        if response.tool_calls:  # крок ще триває: tool call → tools або approval → знову executor
            return {"step_messages": [response]}
        # Крок виконано. До висновку LLM додаємо сирі результати tools: стислий висновок губить дані
        # (напр., інші готелі зі списку), і тоді replanner перепланував би повторний пошук
        tool_data = [f"{m.name} → {m.content}" for m in state["step_messages"] if m.type == "tool"]
        result = (f"Крок {k + 1} ({state['plan'][k]}): {normalize(response.text)}"
                  + "".join(f"\n  [{t}]" for t in tool_data))
        return {"current_step": k + 1, "results": [*state["results"], result],
                "step_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    def route_executor(state: PlanExecuteState) -> Literal["tools", "approval", "replanner"]:
        calls = state["step_messages"][-1].tool_calls if state["step_messages"] else []
        if not calls:
            return "replanner"
        return "approval" if calls[0]["name"] in RISKY_TOOLS else "tools"

    def replanner(state: PlanExecuteState) -> dict:
        d = cast(ReplanDecision, replanner_llm.invoke([SystemMessage(REPLANNER_PROMPT), HumanMessage(progress(state))]))
        d = d.model_copy(update={"reasoning": normalize(d.reasoning),
                                 "updated_steps": d.updated_steps and [normalize(s) for s in d.updated_steps],
                                 "final_answer": d.final_answer and normalize(d.final_answer)})
        k, plan = state["current_step"], state["plan"]
        if d.action == "replan" and d.updated_steps:
            # виконані кроки лишаються в історії, замінюємо лише майбутні
            plan = plan[:k] + d.updated_steps[:MAX_REPLAN_STEPS]
        out_of_steps = k >= len(plan) or k >= MAX_STEPS
        update: dict = {"plan": plan, "decision": d.model_dump(), "completed": d.action == "finish" or out_of_steps}
        if update["completed"]:
            answer = d.final_answer if d.action == "finish" and d.final_answer else None
            if not answer:  # LLM не завершила сама: кроки скінчились або досягнуто MAX_STEPS
                limit = f"Досягнуто ліміт у {MAX_STEPS} кроків. " if k >= MAX_STEPS else ""
                answer = f"{limit}Результати виконаних кроків:\n" + "\n".join(state["results"])
            update["messages"] = [AIMessage(answer)]
        return update

    # pyrefly: ignore[bad-specialization]  # хибне спрацювання Pyrefly: TypedDict-стан підходить LangGraph
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", ToolNode(TOOLS, messages_key="step_messages", handle_tool_errors=True))
    graph.add_node("approval", approval_node)  # переходи — через Command (tools, executor або знову approval)
    graph.add_node("replanner", replanner)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_executor)
    graph.add_edge("tools", "executor")
    graph.add_conditional_edges("replanner", lambda s: END if s["completed"] else "executor", ["executor", END])
    return graph


def open_checkpointer(path: Path | str = DB_PATH) -> SqliteSaver:
    # Файл, а не ":memory:" — стан переживає перезапуск процесу.
    # check_same_thread=False: LangGraph може звертатися до з'єднання з різних потоків.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_app(checkpointer: SqliteSaver | None = None, llm=None):
    """Скомпільований граф. Checkpointer обов'язковий: без нього не працюють ні відновлення, ні interrupt()."""
    return build_graph(llm).compile(checkpointer=checkpointer or open_checkpointer())


def thread_config(thread_id: str) -> RunnableConfig:
    """thread_id — ключ сесії в SqliteSaver: різні thread_id мають незалежний стан."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def human_decision(state, thread: str) -> Command:
    """Відповідь людини для Command(resume=...). Немає людини (stdin закрито) — виходимо, тред чекає в БД."""
    try:
        return Command(resume=ask_human(state.interrupts[0].value))
    except EOFError:
        sys.exit(f"\nТред {thread!r} чекає підтвердження в agent_state.db. "
                 f"Продовжити: python plan_execute.py --thread {thread}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-and-Execute агент бронювання готелів")
    parser.add_argument("query", nargs="?", help="запит; без нього — продовжити тред з останнього checkpoint")
    parser.add_argument("--thread", default="cli", help="thread_id сесії (за замовчуванням: cli)")
    args = parser.parse_args()

    app, config = build_app(), thread_config(args.thread)
    state = app.get_state(config)
    if args.query:
        payload = {"messages": [HumanMessage(args.query)]}
    elif state.next:  # незавершене виконання: процес упав посеред плану або тред чекає підтвердження людини
        print(f"Відновлено тред {args.thread}: виконано {state.values['current_step']} з "
              f"{len(state.values['plan'])} кроків, далі — {state.next[0]}")
        payload = human_decision(state, args.thread) if state.interrupts else None
    else:
        parser.error(f"тред {args.thread!r} не має незавершеного виконання — передайте запит")

    while True:
        for update in app.stream(payload, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "planner":
                    print("План:", *data["plan"], sep="\n  ")
                elif node == "replanner":
                    print(f"replanner: {data['decision']['action']} — {data['decision']['reasoning']}")
                elif node != "__interrupt__":
                    print(f"→ {node}")
        state = app.get_state(config)
        if not state.interrupts:
            break
        # HITL: граф зупинився на interrupt() — питаємо людину і продовжуємо той самий тред
        payload = human_decision(state, args.thread)
    print("\n" + state.values["messages"][-1].text)


if __name__ == "__main__":
    main()
