"""Unit-тести інструментів (Завдання 1). Запуск: pytest -v test_tools.py

Працюють без API-ключа: tools — звичайні Python-функції на mock-даних.
"""

import pytest
from pydantic import ValidationError

from tools import (
    FREE_CANCEL_HOURS,
    MAX_GUESTS,
    MAX_NIGHTS,
    TOOLS,
    TOURIST_TAX,
    BookHotelInput,
    HotelDetailsInput,
    PriceInput,
    SearchHotelsInput,
    book_hotel,
    calculate_total_price,
    get_hotel_details,
    search_hotels,
)

STAY = {"check_in": "2027-03-10", "check_out": "2027-03-12", "guests": 2}


# ── search_hotels ────────────────────────────────────────────────
def test_search_lviv_skips_sold_out():
    out = search_hotels.invoke({**STAY, "city": "Львів"})  # українська назва → канонічна «Львів»
    assert "LVI-001" in out and "за 2 ноч.: 5800 грн" in out
    assert "LVI-002" in out and "LVI-003" not in out  # Opera Grand — немає вільних номерів
    assert out.index("LVI-001") < out.index("LVI-002")  # сортування за рейтингом


def test_search_counts_rooms_for_big_group():
    out = search_hotels.invoke({"city": "Odesa", "check_in": "2027-07-01", "check_out": "2027-07-04",
                                "guests": 3, "min_rating": 8})
    # ODS-002 дешевший за ніч, але на 3 гостей потрібно 2 номери → дорожчий за весь період
    assert "ODS-001" in out and "за 3 ноч.: 10500 грн" in out
    assert "номерів потрібно: 2 | за 3 ноч.: 12600 грн" in out and "ODS-003" not in out


def test_search_no_results():
    assert "немає готелів" in search_hotels.invoke({**STAY, "city": "Kyiv", "max_price_per_night": 500})


# ── get_hotel_details ────────────────────────────────────────────
def test_details():
    out = get_hotel_details.invoke({"hotel_id": " lvi-002 "})  # strip + upper
    assert "Lviv Green Garden" in out and "Тварини: дозволені" in out and "Скасування: безкоштовне" in out


# ── calculate_total_price ────────────────────────────────────────
def test_price_with_breakfast():
    out = calculate_total_price.invoke({"hotel_id": "ODS-001", "check_in": "2027-07-01", "check_out": "2027-07-04",
                                        "guests": 3, "breakfast": True})
    # 3500×3 ночі + 400×3 гостей×3 + збір 50×3×3
    assert "номери (1 шт.) 10500 грн + сніданки 3600 грн" in out and "РАЗОМ 14550 грн" in out


def test_price_breakfast_not_available():
    out = calculate_total_price.invoke({**STAY, "hotel_id": "KYV-003", "breakfast": True})
    assert "не надається" in out and "РАЗОМ 2800 грн" in out  # 1300×2 + збір 200


# ── book_hotel ───────────────────────────────────────────────────
def test_book_ok_and_deterministic():
    args = {"hotel_id": "KYV-001", "check_in": "2027-02-14", "check_out": "2027-02-16", "guests": 2,
            "breakfast": True, "guest_name": "Іван Петренко", "email": "Ivan@Example.com"}
    out = book_hotel.invoke(args)
    assert "Бронювання підтверджено" in out and "до сплати 8000 грн" in out and "ivan@example.com" in out
    assert out == book_hotel.invoke(args)  # той самий код підтвердження
    code = lambda a: book_hotel.invoke(a).split("Код: ")[1][:9]  # noqa: E731  BK-XXXXXX
    # будь-яка зміна бронювання — новий код, а не перезапис старого
    assert len({code(args), code(args | {"breakfast": False}), code(args | {"guest_name": "Олена Петренко"})}) == 3


def test_book_sold_out():
    out = book_hotel.invoke({**STAY, "hotel_id": "LVI-003", "guest_name": "Іван Петренко", "email": "i@ex.com"})
    assert "Бронювання неможливе" in out


# ── Валідація: некоректні входи відхиляються зрозумілим повідомленням ──
@pytest.mark.parametrize(
    "model, args, msg",
    [
        (SearchHotelsInput, {**STAY, "city": "Paris"}, "не підтримується"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "check_in": "2020-01-01"}, "уже минула"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "check_out": "2027-03-09"}, "пізніше за дату заїзду"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "check_out": "2027-05-10"}, "30 ночей"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "guests": 9}, "від 1 до 8"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "min_rating": 11}, "від 0 до 10"),
        (SearchHotelsInput, {**STAY, "city": "Kyiv", "max_price_per_night": -1}, "більшою за 0"),
        (HotelDetailsInput, {"hotel_id": "XXX-999"}, "не знайдено"),
        (PriceInput, {**STAY, "hotel_id": "NOPE"}, "не знайдено"),
        (BookHotelInput, {**STAY, "hotel_id": "KYV-001", "guest_name": "Іван", "email": "i@ex.com"},
         "ім'я та прізвище"),
        (BookHotelInput, {**STAY, "hotel_id": "KYV-001", "guest_name": "Іван Петренко", "email": "ivan(at)gmail"},
         "Некоректний email"),
    ],
)
def test_validation(model, args, msg):
    with pytest.raises(ValidationError, match=msg):
        model(**args)


# ── Помилка валідації через tool повертається LLM як текст, а не падає ──
def test_tool_error_is_readable():
    from langchain_core.messages import AIMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    # pyrefly: ignore[bad-specialization]  # хибне спрацювання Pyrefly на TypedDict-стані LangGraph
    g = StateGraph(MessagesState)
    g.add_node("tools", ToolNode([search_hotels], handle_tool_errors=True))
    g.add_edge(START, "tools")
    call = {"name": "search_hotels", "args": {**STAY, "city": "Paris"}, "id": "1"}
    msg = g.compile().invoke({"messages": [AIMessage(content="", tool_calls=[call])]})["messages"][-1]
    assert msg.status == "error" and "не підтримується" in msg.content


# ── описи для LLM ────────────────────────────────────────────────
def test_descriptions_follow_constants():
    """Ліміти в описах tools підставляються з констант: зміна константи змінює й опис для LLM."""
    assert f"до {MAX_NIGHTS} ночей, 1–{MAX_GUESTS} гостей" in search_hotels.description
    assert f"збір {TOURIST_TAX} грн/особа/ніч" in calculate_total_price.description
    assert f"до {FREE_CANCEL_HOURS} год до заїзду" in get_hotel_details.description
    assert not any("{" in t.description for t in TOOLS)  # усі плейсхолдери підставлено
