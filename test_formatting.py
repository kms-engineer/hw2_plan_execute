"""Тести єдиного формату тексту LLM (formatting.py). Запуск: pytest -v test_formatting.py"""

from datetime import date

import pytest

from formatting import normalize, violations
from knowledge import CORPUS
from tools import HOTELS, booking_summary, calculate_total_price, get_hotel_details, search_hotels

STAY = {"check_in": "2027-03-10", "check_out": "2027-03-12", "guests": 2}


@pytest.mark.parametrize("raw, expected", [
    ("**Opera Grand (LVI-003)**: з 10 по 12 березня 2027 року", "LVI-003 Opera Grand: 2027-03-10 – 2027-03-12"),
    ("# Rynok Square Hotel — 10–12 березня 2027", "LVI-001 Rynok Square Hotel — 2027-03-10 – 2027-03-12"),
    ("з 2027-02-14 до 2027-02-16, LVI-001 — Rynok Square Hotel", "2027-02-14 – 2027-02-16, LVI-001 Rynok Square Hotel"),
    ("2027-03-10–2027-03-12: 14 550 грн, рейтинг 8,7", "2027-03-10 – 2027-03-12: 14550 грн, рейтинг 8.7"),
    ("з 30 березня по 2 квітня 2027, 01.07.2027", "2027-03-30 – 2027-04-02, 2027-07-01"),
    ("заїзд 2027-03-10, виїзд 12 березня", "заїзд 2027-03-10, виїзд 2027-03-12"),  # рік — з дати того ж місяця
    ("2027-12-30 – 2028-01-02, виїзд 2 січня", "2027-12-30 – 2028-01-02, виїзд 2028-01-02"),
    ("Kyiv, Lviv Green Garden, Odesa", "Київ, LVI-002 Lviv Green Garden, Одеса"),  # назву готелю не чіпаємо
    # випадки, які ламали першу версію нормалізатора
    ("12 березня 2027 разом 7120 грн", "2027-03-12 разом 7120 грн"),
    ("LVI-001 — 12 березня 2027", "LVI-001 — 2027-03-12"),
    ("LVI-001 (Rynok Square Hotel)", "LVI-001 Rynok Square Hotel"),
    ("LVI-001 «Rynok Square Hotel» і «Opera Grand»", "LVI-001 Rynok Square Hotel і LVI-003 Opera Grand"),
    ("LVI-003 Rynok Square Hotel", "LVI-001 Rynok Square Hotel"),  # ID не від тієї назви
    ("до сплати 7,120 грн", "до сплати 7120 грн"),
])
def test_normalize(raw, expected):
    assert violations(raw) and normalize(raw) == expected and not violations(expected)


@pytest.mark.parametrize("corrupted", [
    "2027-03-12азом 7120 грн", "LVI-02027-03-01 – 2027-03-12", "LVI-001 (LVI-001 Rynok Square Hotel)",
    "LVI-003 LVI-001 Rynok Square Hotel",
])
def test_violations_detect_corruption(corrupted):
    assert violations(corrupted)


def canonical_texts() -> list[str]:
    """Тексти, які вже в єдиному форматі: усі виходи tools і всі документи бази правил."""
    texts = [search_hotels.invoke({**STAY, "city": c}) for c in ("Київ", "Львів", "Одеса")]
    for hid in HOTELS:
        texts += [get_hotel_details.invoke({"hotel_id": hid}),
                  calculate_total_price.invoke({**STAY, "hotel_id": hid, "breakfast": True}),
                  booking_summary(hid, date(2027, 3, 10), date(2027, 3, 12), 2, True, "Іван Петренко", "a@b.com")]
    return texts + [text for text, _ in CORPUS.values()]


def test_canonical_text_untouched_and_idempotent():
    for text in canonical_texts():
        assert not violations(text) and normalize(text) == text, text
    sample = "Opera Grand, 7 120 грн, з 10 по 12 березня 2027"
    assert normalize(normalize(sample)) == normalize(sample)
