"""Тести бази правил проживання (Завдання 3). Запуск: pytest -v test_knowledge.py

Пошук потребує OPENAI_API_KEY (ембединги text-embedding-3-large); без ключа ці тести пропускаються.
"""

import os
import re

import pytest
from pydantic import ValidationError

from calibrate_rag import evaluate, rankings
from knowledge import (
    CORPUS,
    DOCUMENTS,
    HOTEL_RULES,
    MAX_RESULTS,
    MIN_QUERY_LEN,
    TOPICS,
    KnowledgeQuery,
    get_collection,
    search_knowledge,
)
from tools import FREE_CANCEL_HOURS, HOTELS, TOURIST_TAX, get_hotel_details

needs_key = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="потрібен OPENAI_API_KEY для ембедингів")


def search(query: str, **kw) -> str:
    return search_knowledge.invoke({"query": query, **kw})


def labels(out: str) -> list[str]:
    """Підписи документів у виході: «Правила мережі · тема» / «Правила готелю ID Назва · тема»."""
    return re.findall(r"^\[([^\]]+)\]", out, re.MULTILINE)


# ── Дані ─────────────────────────────────────────────────────────
def test_documents():
    """Кожен документ — одна відома тема; у мережі одна тема — один документ (інакше перекриття неоднозначне)."""
    network_topics = [topic for topic, _, _ in DOCUMENTS.values()]
    assert len(network_topics) == len(set(network_topics)) and set(network_topics) <= TOPICS.keys()
    assert all(rules.keys() <= TOPICS.keys() for rules in HOTEL_RULES.values())
    assert all(text.endswith(".") for text, _ in CORPUS.values())


FACILITIES = ("паркінг", "ресторан", "басейн", "spa", "спортзал", "бар", "трансфер", "сад", "пляж", "кухн")


def test_hotel_rules_match_hotel_data():
    """Правила готелю не суперечать get_hotel_details: тварини, сніданок, зручності."""
    assert HOTEL_RULES.keys() == HOTELS.keys()
    for hid, rules in HOTEL_RULES.items():
        h, text = HOTELS[hid], " ".join(rules.values())
        assert ("Тварини не допускаються" in rules["тварини"]) == (not h["pets"]), hid
        # без сніданку — явно, інакше загальне правило «шведський стіл 07:00–10:30» нічим не перекрите
        assert ("Сніданків немає" in rules.get("харчування", "")) == (h["breakfast"] is None), hid
        amenities = " ".join(h["amenities"]).lower()
        mentioned = text.lower().replace("паркінгу немає", "")
        assert "паркінгу немає" not in text.lower() or "паркінг" not in amenities, hid
        for word in FACILITIES:  # згадана в правилах зручність є в даних готелю
            assert not re.search(rf"\b{word}", mentioned) or word in amenities, (hid, word)


def test_network_rules_match_tools():
    """Загальні правила не суперечать tools: збір і строк скасування — з констант tools, Wi-Fi усюди."""
    assert f"{TOURIST_TAX} грн за кожного гостя за кожну ніч" in DOCUMENTS["tourist-tax"][2]
    assert f"за {FREE_CANCEL_HOURS} год до заїзду" in DOCUMENTS["cancellation"][2]
    assert all("Wi-Fi" in h["amenities"] for h in HOTELS.values())  # «Wi-Fi працює і під час відключень»


def test_hotel_card_lists_rule_topics():
    """get_hotel_details не дублює текст правил, а каже, з яких тем у готелю свої правила."""
    assert "Особливі правила готелю (search_knowledge з hotel_id): тварини, паркінг." in get_hotel_details.invoke(
        {"hotel_id": "LVI-002"})


def test_description_follows_constants():
    assert f"(1–{MAX_RESULTS})" in search_knowledge.description
    assert f"коротший за {MIN_QUERY_LEN} симв." in search_knowledge.description
    assert all(t in search_knowledge.description for t in TOPICS)
    assert "{" not in search_knowledge.description  # усі плейсхолдери підставлено


# ── Валідація ────────────────────────────────────────────────────
def test_query_validated():
    q = KnowledgeQuery(query="тварини", hotel_id=" lvi-002 ", topics=[" Тварини ", "паркінг", "тварини"])
    assert q.hotel_id == "LVI-002" and q.topics == ["тварини", "паркінг"]  # нормалізовано, без дублів
    for bad in ({"query": "ok"}, {"query": "тварини", "hotel_id": "XXX-999"},
                {"query": "тварини", "topics": ["погода"]}):
        with pytest.raises(ValidationError):
            KnowledgeQuery.model_validate(bad)


# ── Пошук ────────────────────────────────────────────────────────
@needs_key
def test_collection_synced():
    assert sorted(get_collection().get()["ids"]) == sorted(CORPUS)  # без застарілих документів


@needs_key
def test_hotel_filter_and_override():
    """Правила готелю замінюють загальні правила тієї ж теми; правил інших готелів у виході немає."""
    out = search("чи можна з собакою і скільки це коштує", hotel_id="KYV-002")
    assert labels(out) == ["Правила готелю KYV-002 Podil Boutique · тварини"] and "20 кг" in out
    assert "до 10 кг" not in out  # загальне правило про тварин перекрите
    out = search("о котрій сніданок", hotel_id="KYV-003")
    assert labels(out)[0] == "Правила готелю KYV-003 Kyiv Budget Inn · харчування"
    assert "Сніданків немає" in out and "шведський стіл" not in out


@needs_key
@pytest.mark.parametrize("hotel_id, topic", [(h, t) for h, rules in HOTEL_RULES.items() for t in rules])
def test_every_hotel_rule_overrides_network(hotel_id, topic):
    """Для кожного правила готелю: пошук з цим готелем і темою повертає саме його, без загального правила теми."""
    out = search("правила проживання", hotel_id=hotel_id, topics=[topic])
    assert labels(out) == [f"Правила готелю {hotel_id} {HOTELS[hotel_id]['name']} · {topic}"]
    assert HOTEL_RULES[hotel_id][topic] in out


@needs_key
def test_topics_return_all_named_topics():
    """З topics кількість результатів = кількість тем: короткий запит не губить жодну з них."""
    out = search("о котрій заселення і який депозит", hotel_id="ODS-003", topics=["заселення", "депозит"])
    assert sorted(labels(out)) == ["Правила готелю ODS-003 Black Sea Hostel · депозит",
                                   "Правила готелю ODS-003 Black Sea Hostel · заселення"]
    assert "15:00" in out and "200 грн" in out and "14:00" not in out and "500 грн" not in out
    out = search("о котрій заселення і які документи", topics=["заселення", "документи"])
    assert sorted(labels(out)) == ["Правила мережі · документи", "Правила мережі · заселення"]
    # тема без жодного правила (паркінг є лише в готелів) — прямо сказано, а не мовчки пропущено
    out = search("о котрій заселення і де паркуватися", topics=["заселення", "паркінг"])
    assert labels(out) == ["Правила мережі · заселення"] and "Правил на теми паркінг немає" in out


@needs_key
def test_irrelevant_query_says_so():
    assert search("який курс біткоїна сьогодні").startswith("Релевантних правил не знайдено")


@needs_key
@pytest.mark.parametrize("query, topic", [
    ("правила проживання з тваринами", "тварини"),
    ("час заїзду та виїзду", "заселення"),
    ("які документи потрібні для заселення", "документи"),
    ("чи повернуть гроші, якщо скасувати бронювання", "скасування"),
    ("що робити під час повітряної тривоги", "безпека"),
    ("як доїхати потягом до Львова", "транспорт"),
])
def test_semantic_search_without_topics(query, topic):
    """Без topics — лише найближча тема: на вузьке питання 1–2 документи, а не фіксований топ-N."""
    found = labels(search(query))
    assert found[0] == f"Правила мережі · {topic}" and len(found) <= 2


@needs_key
def test_threshold_quality():
    """Поріг пошуку без topics не гірший за відкалібрований рівень на розміченому наборі (calibrate_rag.py)."""
    precision, recall, exact = evaluate(rankings())
    assert precision >= 0.8 and recall >= 0.9 and exact >= 23
