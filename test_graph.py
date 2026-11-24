"""Офлайн-тести графа Plan-and-Execute, persistence і HITL на «заскриптованій» fake-LLM (без API-ключа).

Запуск: pytest -v test_graph.py
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from plan_execute import (
    MAX_PLAN_STEPS,
    MAX_REPLAN_STEPS,
    Plan,
    ReplanDecision,
    build_app,
    open_checkpointer,
    thread_config,
)
from tools import BOOKINGS

STAY = {"check_in": "2027-03-10", "check_out": "2027-03-12", "guests": 2}
BOOK = {**STAY, "hotel_id": "LVI-001", "guest_name": "Іван Петренко", "email": "ivan@example.com"}
CONTINUE = ReplanDecision(reasoning="далі за планом", action="continue")


class ScriptedLLM:
    """Fake-LLM: віддає задані відповіді по черзі — Plan (planner), AIMessage (executor), ReplanDecision (replanner)."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def _next(self, _):
        return self.outputs.pop(0)

    def with_structured_output(self, schema):
        return RunnableLambda(self._next)

    def bind_tools(self, tools, **kwargs):
        return RunnableLambda(self._next)


def call(name, args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


def finish(answer):
    return ReplanDecision(reasoning="ціль досягнута", action="finish", final_answer=answer)


def run_nodes(app, payload, config):
    """Назви вузлів у порядку виконання."""
    return [node for update in app.stream(payload, config, stream_mode="updates") for node in update]


@pytest.fixture
def make_app(tmp_path):
    db = tmp_path / "state.db"
    return lambda *outputs: build_app(open_checkpointer(db), ScriptedLLM(*outputs))


# ── Завдання 1: planner → executor ⇄ tools → replanner ───────────
def test_plan_execute_cycle(make_app):
    app = make_app(
        Plan(goal="Готелі у Львові", steps=["search_hotels: Львів", "відповісти користувачу"]),
        call("search_hotels", {**STAY, "city": "Lviv"}), AIMessage("Знайдено LVI-001 і LVI-002"), CONTINUE,
        AIMessage("Найкращий — LVI-001"), finish("LVI-001, 5800 грн за 2 ночі"),
    )
    cfg = thread_config("t")
    assert run_nodes(app, {"messages": [HumanMessage("Готелі у Львові?")]}, cfg) == [
        "planner", "executor", "tools", "executor", "replanner", "executor", "replanner"]
    s = app.get_state(cfg).values
    assert s["completed"] and s["current_step"] == 2 and len(s["results"]) == 2
    assert "LVI-001 Rynok Square Hotel |" in s["results"][0]  # сирий результат tool збережено в результаті кроку
    assert s["messages"][-1].content == "LVI-001, 5800 грн за 2 ночі" and s["step_messages"] == []


def test_replan_replaces_only_remaining_steps(make_app):
    app = make_app(
        Plan(goal="Opera Grand", steps=["get_hotel_details: LVI-003", "calculate_total_price: LVI-003"]),
        AIMessage("LVI-003: вільних номерів 0"),
        ReplanDecision(reasoning="немає номерів", action="replan", updated_steps=["search_hotels: альтернатива"]),
        AIMessage("Альтернатива — LVI-001"), finish("Opera Grand зайнятий, альтернатива — LVI-001"),
    )
    cfg = thread_config("t")
    app.invoke({"messages": [HumanMessage("Хочу в Opera Grand")]}, cfg)
    s = app.get_state(cfg).values
    assert s["plan"] == ["get_hotel_details: LVI-003", "search_hotels: альтернатива"]  # виконаний крок лишився
    assert s["current_step"] == 2 and s["decision"]["action"] == "finish"


def test_plan_and_replan_are_capped(make_app):
    """Planner не дає більше MAX_PLAN_STEPS кроків, replan — більше MAX_REPLAN_STEPS нових."""
    app = make_app(
        Plan(goal="g", steps=[f"крок {i}" for i in range(MAX_PLAN_STEPS + 3)]), AIMessage("A"),
        ReplanDecision(reasoning="інакше", action="replan", updated_steps=[f"новий {i}" for i in range(9)]),
        *[x for _ in range(MAX_REPLAN_STEPS) for x in (AIMessage("ок"), CONTINUE)],
    )
    cfg = thread_config("caps")
    app.invoke({"messages": [HumanMessage("q")]}, cfg)
    plan = app.get_state(cfg).values["plan"]
    assert plan == ["крок 0", *[f"новий {i}" for i in range(MAX_REPLAN_STEPS)]]


# ── Завдання 2: SqliteSaver ──────────────────────────────────────
def test_resume_after_crash(tmp_path):
    db, cfg = tmp_path / "state.db", thread_config("crash")
    app1 = build_app(open_checkpointer(db), ScriptedLLM(Plan(goal="g", steps=["a", "b"]), AIMessage("A"), CONTINUE))
    for update in app1.stream({"messages": [HumanMessage("q")]}, cfg, stream_mode="updates", durability="sync"):
        if "replanner" in update:
            break  # «падіння» після 1-го кроку
    # «Новий процес»: нове з'єднання з тим самим файлом. У fake-LLM лише відповіді для решти плану:
    # якби planner запустився знову, він забрав би не той об'єкт і тест упав би
    app2 = build_app(open_checkpointer(db), ScriptedLLM(AIMessage("B"), finish("A і B")))
    state = app2.get_state(cfg)
    assert state.values["current_step"] == 1 and state.next == ("executor",)
    assert run_nodes(app2, None, cfg) == ["executor", "replanner"]
    assert app2.get_state(cfg).values["messages"][-1].content == "A і B"


def test_threads_are_independent(make_app):
    app = make_app(Plan(goal="g", steps=["a"]), AIMessage("A"), finish("відповідь A"))
    app.invoke({"messages": [HumanMessage("q")]}, thread_config("A"))
    assert app.get_state(thread_config("A")).values["messages"][-1].content == "відповідь A"
    assert app.get_state(thread_config("B")).values == {}  # інший thread_id — порожній стан


# ── Завдання 4: HITL ─────────────────────────────────────────────
def booking_flow(make_app, decisions, *after):
    """Граф зупиняється перед book_hotel, людина по черзі відповідає decisions.

    Повертає результат кроку й summary кожного запиту підтвердження."""
    BOOKINGS.clear()
    app = make_app(Plan(goal="g", steps=["book_hotel: LVI-001"]), call("book_hotel", BOOK), *after)
    cfg = thread_config("hitl")
    result, summaries = app.invoke({"messages": [HumanMessage("Забронюй LVI-001")]}, cfg), []
    for decision in decisions:
        summaries.append(result["__interrupt__"][0].value["summary"])  # деталі дії, які бачить людина
        assert app.get_state(cfg).next == ("approval",) and not BOOKINGS  # до відповіді нічого не заброньовано
        result = app.invoke(Command(resume=decision), cfg)
    assert "__interrupt__" not in result and "до сплати 6000 грн" in summaries[0]
    return app.get_state(cfg).values["results"][0], summaries


def test_hitl_approve(make_app):
    result, _ = booking_flow(make_app, [{"action": "approve"}], AIMessage("Заброньовано"), finish("ok"))
    assert "Бронювання підтверджено! Код: BK-" in result and len(BOOKINGS) == 1


def test_hitl_reject(make_app):
    result, _ = booking_flow(make_app, [{"action": "reject", "reason": "Клієнт передумав"}],
                             AIMessage("Оператор відхилив"), finish("Не заброньовано"))
    assert "Оператор ВІДХИЛИВ book_hotel: Клієнт передумав" in result and not BOOKINGS


def test_hitl_edit_reconfirmed(make_app):
    """edit не виконує одразу: людина ще раз бачить нову суму й підтверджує саме її."""
    result, summaries = booking_flow(make_app, [{"action": "edit", "args": {"breakfast": True}}, {"action": "approve"}],
                                     AIMessage("Заброньовано зі сніданком"), finish("ok"))
    assert "сніданок: так" in summaries[1] and "до сплати 7120 грн" in summaries[1]  # 5800 + сніданки 1120 + збір 200
    assert "сніданок: так" in result and "до сплати 7120 грн" in result and len(BOOKINGS) == 1


def test_hitl_edit_then_reject(make_app):
    result, summaries = booking_flow(make_app, [{"action": "edit", "args": {"guests": 4}}, {"action": "reject"}],
                                     AIMessage("Оператор відхилив"), finish("Не заброньовано"))
    assert "номерів: 2" in summaries[1] and "Оператор ВІДХИЛИВ" in result and not BOOKINGS


def test_sold_out_skips_human(make_app):
    BOOKINGS.clear()
    app = make_app(Plan(goal="g", steps=["book_hotel"]), call("book_hotel", {**BOOK, "hotel_id": "LVI-003"}),
                   AIMessage("Немає номерів"), finish("Opera Grand заброньований повністю"))
    cfg = thread_config("t")
    assert "__interrupt__" not in app.invoke({"messages": [HumanMessage("Забронюй LVI-003")]}, cfg)
    assert "Бронювання неможливе" in app.get_state(cfg).values["results"][0] and not BOOKINGS


def test_invalid_booking_skips_human(make_app):
    app = make_app(Plan(goal="g", steps=["book_hotel"]), call("book_hotel", {**BOOK, "email": "bad"}),
                   AIMessage("Некоректний email"), finish("Потрібен коректний email"))
    cfg = thread_config("t")
    assert "__interrupt__" not in app.invoke({"messages": [HumanMessage("Забронюй")]}, cfg)
    assert "Некоректний email" in app.get_state(cfg).values["results"][0]  # помилку валідації отримала LLM
