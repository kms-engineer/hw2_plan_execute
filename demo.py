"""Демонстрації всіх завдань ДЗ2 на реальному LLM з автоперевірками → demo_results.json.

Запуск:  python demo.py                  # усі групи
         python demo.py hitl rag         # вибрані: plan, rag, persist, threads, hitl
Кожен сценарій має свій thread_id в agent_state.db і перед запуском очищає його (відтворюваність).
"""

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from formatting import violations
from knowledge import KnowledgeQuery
from plan_execute import build_app, thread_config
from tools import BOOKINGS, BookHotelInput, HotelDetailsInput, PriceInput, SearchHotelsInput

RESULTS_PATH = Path(__file__).parent / "demo_results.json"
BOOK_QUERY = ("Забронюй Rynok Square Hotel (LVI-001) з 10 по 12 березня 2027 на двох на ім'я Іван Петренко, "
              "email ivan.petrenko@example.com")
PERSIST_QUERY = ("Знайди найдешевший готель в Одесі з 1 по 4 липня 2027 на трьох з рейтингом від 8, порахуй повну "
                 "вартість зі сніданком і скажи, які документи потрібні для заселення")


# ── Запуск і збір траєкторії ─────────────────────────────────────
SCHEMAS: dict[str, type[BaseModel]] = {"search_hotels": SearchHotelsInput, "get_hotel_details": HotelDetailsInput,
                                       "calculate_total_price": PriceInput, "book_hotel": BookHotelInput,
                                       "search_knowledge": KnowledgeQuery}


def normalized(name: str, args: dict) -> dict:
    """Аргументи так, як їх отримує tool після Pydantic: дати YYYY-MM-DD, канонічні місто й ID, усі поля."""
    try:
        return SCHEMAS[name].model_validate(args).model_dump(mode="json")
    except ValidationError:
        return args  # невалідні — як є: tool поверне помилку валідації (status=error)


def collect(stream, trace: dict) -> None:
    """Розібрати потік оновлень вузлів (stream_mode="updates") у траєкторію.

    Кожен виклик tool — один запис {id, name, args, status, result}: tool call з executor (або змінений людиною
    в approval) зіставляється з його ToolMessage за id.
    """
    for update in stream:
        for node, data in update.items():
            trace["nodes"].append(node)
            if node == "__interrupt__":
                trace["interrupts"].append(data[0].value)
                continue
            data = data or {}  # approval з approve (або без запиту людині) повертає лише goto, без оновлення стану
            if node == "planner":
                trace["plans"].append(data["plan"])
            elif node == "replanner":
                trace["decisions"].append({k: data["decision"][k] for k in ("action", "reasoning", "updated_steps")})
            calls = {c["id"]: c for c in trace["tool_calls"]}
            for m in data.get("step_messages", []):
                if m.type == "ai":
                    for c in m.tool_calls:  # edit в approval: той самий id, нові args — оновлюємо запис
                        if c["id"] not in calls:
                            calls[c["id"]] = {"id": c["id"], "name": c["name"], "args": {}, "status": None,
                                              "result": None}
                            trace["tool_calls"].append(calls[c["id"]])
                        calls[c["id"]]["args"] = normalized(c["name"], c["args"])
                elif m.type == "tool":
                    calls[m.tool_call_id] |= {"status": m.status, "result": m.content}


def new_trace(thread: str, query: str | None) -> dict:
    return {"thread": thread, "query": query, "plans": [], "decisions": [], "tool_calls": [], "interrupts": [],
            "nodes": []}


def run(app, thread: str, query: str, decisions: Sequence[dict] = (), fresh: bool = True) -> dict:
    """Виконати запит у треді; на кожен interrupt() подати наступне рішення людини з decisions."""
    config = thread_config(thread)
    if fresh:
        app.checkpointer.delete_thread(thread)
    trace, pending = new_trace(thread, query), list(decisions)
    payload, t0 = {"messages": [HumanMessage(query)]}, time.monotonic()
    while True:
        collect(app.stream(payload, config, stream_mode="updates"), trace)
        state = app.get_state(config)
        if not state.interrupts:
            break
        # Повторний запит підтвердження, якого сценарій не чекав (напр., LLM повторила book_hotel) — відхиляємо
        payload = Command(resume=pending.pop(0) if pending else {"action": "reject", "reason": "непередбачений запит"})
    trace |= {"final_answer": state.values["messages"][-1].text, "steps_done": state.values["current_step"],
              "time_s": round(time.monotonic() - t0, 1)}
    return trace


def used(trace: dict) -> list[str]:
    return [c["name"] for c in trace["tool_calls"]]



def has(text: str, *needles: str) -> bool:
    """Чи є всі needles у тексті без урахування пробілів («7 120» == «7120»)."""
    squash = lambda s: re.sub(r"[\s  ]", "", s)  # noqa: E731
    return all(squash(n) in squash(text) for n in needles)


def record(sid: str, task: int, trace: dict, checks: dict[str, bool]) -> dict:
    # Ті самі правила, що й у formatting.normalize: весь текст LLM у записі — план, рішення, відповідь
    texts = [trace["final_answer"], *(s for plan in trace["plans"] for s in plan),
             *(d["reasoning"] for d in trace["decisions"]),
             *(s for d in trace["decisions"] for s in d["updated_steps"] or [])]
    checks |= {"єдиний формат у плані, рішеннях і відповіді (formatting.violations)": not any(map(violations, texts))}
    ok = all(checks.values())
    print(f"\n{'✅' if ok else '❌'} {sid} ({trace['time_s']} с, кроків {trace['steps_done']}): {trace['query']}")
    for i, plan in enumerate(trace["plans"]):
        print("   план:" if i == 0 else "   новий план:", *plan, sep="\n     · ")
    for d in trace["decisions"]:
        print(f"   replanner: {d['action']} — {d['reasoning']}")
    print(f"   tools: {', '.join(used(trace)) or '—'}\n   відповідь: {trace['final_answer'][:300]}")
    for name, passed in checks.items():
        if not passed:
            print(f"   ✗ {name}")
    return {"id": sid, "task": task, **trace, "checks": checks, "status": "success" if ok else "failed"}


# ── Завдання 1: Plan-and-Execute ─────────────────────────────────
def demo_plan(app) -> list[dict]:
    t1 = run(app, "plan-simple", "Які готелі є в Одесі з 1 по 4 липня 2027 на трьох з рейтингом від 8?")
    t2 = run(app, "plan-medium", "Знайди готель у Львові з 10 по 12 березня 2027 на двох з найвищим рейтингом "
                                 "і порахуй повну вартість зі сніданком")
    t3 = run(app, "plan-replan", "Знайди найдешевший готель у Києві на 14–16 лютого 2027 на двох і порахуй "
                                 "повну вартість зі сніданком")
    return [
        record("PLAN-SIMPLE", 1, t1, {
            "план з 1–2 кроків": len(t1["plans"][0]) <= 2,
            "викликано search_hotels": "search_hotels" in used(t1),
            "у відповіді ODS-001 (єдиний з рейтингом ≥ 8 і місцями на трьох в 1 номері)":
                has(t1["final_answer"], "ODS-001"),
        }),
        record("PLAN-MEDIUM", 1, t2, {
            "search_hotels → calculate_total_price":
                {"search_hotels", "calculate_total_price"} <= set(used(t2))
                and used(t2).index("search_hotels") < used(t2).index("calculate_total_price"),
            "LVI-001 і 7120 грн (Opera Grand з вищим рейтингом зайнятий)": has(t2["final_answer"], "LVI-001", "7120"),
        }),
        record("PLAN-REPLAN", 1, t3, {
            "replanner змінив план (replan)": any(d["action"] == "replan" for d in t3["decisions"]),
            "KYV-001 і 8000 грн (у найдешевшого KYV-003 немає сніданку)": has(t3["final_answer"], "KYV-001", "8000"),
        }),
    ]


# ── Завдання 3: Agentic RAG ──────────────────────────────────────
HOTEL_TOOLS = {"search_hotels", "get_hotel_details", "calculate_total_price", "book_hotel"}


def demo_rag(app) -> list[dict]:
    k = run(app, "rag-knowledge", "О котрій заселення й виселення і які документи потрібні для заселення?")
    t = run(app, "rag-tools", "Порахуй повну вартість Rynok Square Hotel (LVI-001) з 10 по 12 березня 2027 "
                              "на двох зі сніданком")
    m = run(app, "rag-mixed", "Їду до Львова з собакою з 10 по 12 березня 2027 на двох. Знайди готель, де можна "
                              "з твариною, і поясни правила проживання з тваринами")
    o = run(app, "rag-override", "О котрій заселення й виселення і який депозит у Black Sea Hostel (ODS-003)?")
    policies = lambda t, hid: [c for c in t["tool_calls"]  # noqa: E731
                               if c["name"] == "search_knowledge" and c["args"].get("hotel_id") == hid]
    hostel = policies(o, "ODS-003")
    return [
        record("RAG-KNOWLEDGE", 3, k, {
            "search_knowledge без hotel_id (стандартні правила мережі, готель не уточнювали)": bool(policies(k, None)),
            "жодного tool готелів (правила — лише з бази знань)": not HOTEL_TOOLS & set(used(k)),
            "відповідь з бази: 14:00, 12:00, паспорт": has(k["final_answer"], "14:00", "12:00", "паспорт"),
        }),
        record("RAG-TOOLS", 3, t, {
            "викликано calculate_total_price": "calculate_total_price" in used(t),
            "search_knowledge НЕ викликано (потрібне обчислення, а не правила)": "search_knowledge" not in used(t),
            "7120 грн": has(t["final_answer"], "7120"),
        }),
        record("RAG-MIXED", 3, m, {
            "tool готелів для пошуку": bool({"search_hotels", "get_hotel_details"} & set(used(m))),
            "search_knowledge з hotel_id=LVI-002 (фільтр за метаданими)": bool(policies(m, "LVI-002")),
            # «200 грн» саме по собі може бути й туристичним збором (50 × 2 гостей × 2 ночі) — тому перевіряємо,
            # що search_knowledge повернув правило LVI-002 замість загального, а відповідь його цитує
            "правило LVI-002 перекрило загальне (у виході search_knowledge — правила готелю, а не мережі)": bool(
                policies(m, "LVI-002")) and all("Правила готелю LVI-002" in c["result"] and "Правила мережі · тварини"
                                                not in c["result"] for c in policies(m, "LVI-002")),
            "відповідь: LVI-002 і доплата 200 грн за тварину (у мережі — 300 грн)": has(
                m["final_answer"], "LVI-002", "200 грн") and not re.search(r"(?<!\d)300 грн", m["final_answer"]),
        }),
        record("RAG-OVERRIDE", 3, o, {
            "search_knowledge з hotel_id=ODS-003": bool(hostel),
            "у виході правила готелю замість загальних (немає 14:00 і депозиту 500 грн)": bool(hostel) and all(
                "14:00" not in c["result"] and "500 грн" not in c["result"] for c in hostel),
            "відповідь: 15:00, 11:00, депозит 200 грн": has(o["final_answer"], "15:00", "11:00", "200 грн"),
        }),
    ]


# ── Завдання 2: відновлення після падіння процесу ────────────────
def crash_child(thread: str) -> None:
    """Процес №1: виконує план і «падає» (os._exit) одразу після першого кроку."""
    app, config = build_app(), thread_config(thread)
    trace = new_trace(thread, PERSIST_QUERY)
    # durability="sync": checkpoint записується до початку наступного вузла — раптове падіння не губить крок
    for update in app.stream({"messages": [HumanMessage(PERSIST_QUERY)]}, config, stream_mode="updates",
                             durability="sync"):
        collect([update], trace)
        if "replanner" in update:
            print(json.dumps(trace, ensure_ascii=False), flush=True)
            os._exit(1)  # без cleanup і закриття БД — як kill -9 або падіння сервера


def resume_child(thread: str) -> None:
    """Процес №2: новий процес, нове з'єднання з agent_state.db — відновлює стан і продовжує план."""
    app, config = build_app(), thread_config(thread)
    s = app.get_state(config)
    restored = {"current_step": s.values["current_step"], "plan": s.values["plan"], "results": s.values["results"],
                "next": list(s.next), "checkpoint_id": s.config.get("configurable", {})["checkpoint_id"]}
    trace = new_trace(thread, None)
    collect(app.stream(None, config, stream_mode="updates", durability="sync"), trace)  # None = продовжити
    s = app.get_state(config)
    print(json.dumps(trace | {"restored": restored, "final_answer": s.values["messages"][-1].text,
                              "steps_done": s.values["current_step"], "completed": s.values["completed"]},
                     ensure_ascii=False))


def demo_persist(app) -> list[dict]:
    thread = "persist-crash"
    app.checkpointer.delete_thread(thread)
    t0 = time.monotonic()
    procs = [subprocess.run([sys.executable, __file__, cmd, thread], capture_output=True, text=True,
                            cwd=Path(__file__).parent) for cmd in ("_crash", "_resume")]
    crash, resume = (json.loads(p.stdout.strip().splitlines()[-1]) for p in procs)
    # Одна траєкторія на два процеси — у тій самій схемі, що й решта сценаріїв
    trace = {"thread": thread, "query": PERSIST_QUERY, "plans": crash["plans"],
             **{k: crash[k] + resume[k] for k in ("decisions", "tool_calls", "interrupts", "nodes")},
             "final_answer": resume["final_answer"], "steps_done": resume["steps_done"],
             "time_s": round(time.monotonic() - t0, 1),
             "crash": {"exit_code": procs[0].returncode, "after_nodes": len(crash["nodes"])},
             "restored": resume["restored"]}
    checks = {
        "процес №1 упав (код 1) після 1-го кроку": procs[0].returncode == 1 and crash["nodes"].count("replanner") == 1,
        "процес №2 відновив current_step=1 і той самий план": resume["restored"]["current_step"] == 1
                                                             and resume["restored"]["plan"] == crash["plans"][0],
        "планер не запускався вдруге": "planner" not in resume["nodes"],
        "план завершено: ODS-001, 14550 грн і документи": resume["completed"]
                                                        and has(resume["final_answer"], "ODS-001", "14550", "паспорт"),
    }
    out = record("PERSIST-CRASH", 2, trace, checks)
    print(f"   процес №1: {' → '.join(crash['nodes'])} → 💥 os._exit({procs[0].returncode})\n"
          f"   процес №2 get_state: крок {resume['restored']['current_step']}/{len(resume['restored']['plan'])}, "
          f"next={resume['restored']['next']}\n   процес №2: {' → '.join(resume['nodes'])}")
    return [out]


# ── Завдання 2: незалежність thread_id (пам'ять сесії) ───────────
def demo_threads(app) -> list[dict]:
    a1 = run(app, "user-A", "Знайди готель у Львові з 10 по 12 березня 2027 на двох з найвищим рейтингом")
    a2 = run(app, "user-A", "Скільки він коштуватиме зі сніданком?", fresh=False)  # той самий тред
    b = run(app, "user-B", "Скільки він коштуватиме зі сніданком?")  # інший тред — без контексту
    conn = app.checkpointer.conn
    counts = dict(conn.execute("SELECT thread_id, COUNT(*) FROM checkpoints WHERE thread_id IN ('user-A', 'user-B') "
                               "GROUP BY thread_id").fetchall())
    msgs = {t: len(app.get_state(thread_config(t)).values["messages"]) for t in ("user-A", "user-B")}
    print(f"\n   checkpoints у agent_state.db: {counts}; повідомлень у діалозі: {msgs}")
    calc_a = [c["args"].get("hotel_id") for c in a2["tool_calls"] if c["name"] == "calculate_total_price"]
    return [
        record("THREAD-A-1", 2, a1, {"знайдено LVI-001": has(a1["final_answer"], "LVI-001")}),
        record("THREAD-A-2", 2, a2, {
            "«він» = LVI-001 з попередньої відповіді треду": "LVI-001" in calc_a,
            "7120 грн": has(a2["final_answer"], "7120"),
        }),
        record("THREAD-B", 2, b, {
            "тред B не знає про LVI-001": not has(b["final_answer"], "LVI-001"),
            "calculate_total_price не викликано — агент уточнює готель": "calculate_total_price" not in used(b),
            "діалог A: 4 повідомлення, B: 2": msgs == {"user-A": 4, "user-B": 2},
        }) | {"checkpoints": counts},
    ]


# ── Завдання 4: HITL ─────────────────────────────────────────────
SOLD_OUT_QUERY = ("Забронюй Opera Grand (LVI-003) з 10 по 12 березня 2027 на двох на ім'я Іван Петренко, "
                  "email ivan.petrenko@example.com")


def demo_hitl(app) -> list[dict]:
    out = []
    for sid, query, decisions in [
        ("HITL-APPROVE", BOOK_QUERY, [{"action": "approve"}]),
        ("HITL-REJECT", BOOK_QUERY, [{"action": "reject", "reason": "Клієнт передумав"}]),
        # edit не виконує одразу: людина бачить нову суму й підтверджує ще раз
        ("HITL-EDIT", BOOK_QUERY, [{"action": "edit", "args": {"breakfast": True}}, {"action": "approve"}]),
        ("HITL-SOLDOUT", SOLD_OUT_QUERY, []),  # у LVI-003 немає номерів — людину не питають
    ]:
        BOOKINGS.clear()
        t = run(app, sid.lower(), query, decisions)
        booked = [c["result"] for c in t["tool_calls"] if c["name"] == "book_hotel" and c["status"] == "success"
                  and "Бронювання підтверджено" in c["result"]]
        summaries = [i["summary"] for i in t["interrupts"]]
        first_6000 = bool(summaries) and has(summaries[0], "6000грн")
        if sid == "HITL-APPROVE":
            checks = {"interrupt() з деталями: 1 запит, сума 6000 грн": len(summaries) == 1 and first_6000,
                      "book_hotel виконано": len(booked) == 1 and len(BOOKINGS) == 1,
                      "код BK- у відповіді": "BK-" in t["final_answer"]}
        elif sid == "HITL-REJECT":
            checks = {"interrupt() з деталями: 1 запит, сума 6000 грн": len(summaries) == 1 and first_6000,
                      "book_hotel НЕ виконано": not booked and not BOOKINGS,
                      "агент не повторював бронювання": used(t).count("book_hotel") == 1,
                      "відповідь пояснює причину відмови": "передумав" in t["final_answer"]}
        elif sid == "HITL-EDIT":
            checks = {"2 запити: спершу 6000 грн, після edit — нова сума 7120 грн зі сніданком": len(summaries) == 2
                      and first_6000 and has(summaries[1], "сніданок: так", "7120грн"),
                      "виконано саме підтверджене (сніданок, 7120 грн)": len(booked) == 1
                      and has(booked[0], "сніданок: так", "7120грн")}
        else:
            sold_out = [c for c in t["tool_calls"] if c["name"] == "book_hotel" and c["args"]["hotel_id"] == "LVI-003"]
            # Агент може або спробувати book_hotel (tool відмовить без запиту людині), або перевірити наявність
            # через search_hotels і не бронювати взагалі — обидва шляхи правильні
            checks = {"людину не питали про LVI-003 (номерів немає)": not any(
                          i["args"]["hotel_id"] == "LVI-003" for i in t["interrupts"]),
                      "book_hotel для LVI-003 відмовив або не викликався": all(
                          "Бронювання неможливе" in c["result"] for c in sold_out),
                      "нічого не заброньовано": not booked and not BOOKINGS,
                      "відповідь пояснює, що номерів немає": has(t["final_answer"], "LVI-003") and any(
                          w in t["final_answer"].lower() for w in ("немає", "недостатньо", "зайнят", "неможлив",
                                                                    "не вдалося"))}
        out.append(record(sid, 4, t | {"human_decisions": decisions}, checks))
        for summary, decision in zip(summaries, decisions):
            print(f"   ⏸ interrupt: {summary}\n   👤 рішення: {decision}")
    return out


GROUPS = {"plan": demo_plan, "rag": demo_rag, "persist": demo_persist, "threads": demo_threads, "hitl": demo_hitl}


def main(groups: list[str]) -> None:
    app = build_app()
    # Перезапуск частини груп оновлює лише їхні записи у файлі результатів
    results = {r["id"]: r for r in json.loads(RESULTS_PATH.read_text())} if RESULTS_PATH.exists() else {}
    for g in groups:
        print(f"\n═══ {g} ═══")
        results |= {r["id"]: r for r in GROUPS[g](app)}
    RESULTS_PATH.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=2))
    ok = sum(r["status"] == "success" for r in results.values())
    print(f"\nУспішно: {ok}/{len(results)} → {RESULTS_PATH.name}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["_crash"]:
        crash_child(sys.argv[2])
    elif sys.argv[1:2] == ["_resume"]:
        resume_child(sys.argv[2])
    else:
        main(sys.argv[1:] or list(GROUPS))
