"""Єдиний формат даних у тексті LLM: правила (violations) і детерміноване виправлення (normalize).

Промпт (FORMAT_RULES) просить LLM писати в єдиному форматі, але виконує це з певною ймовірністю. Тому кожен текст
LLM, що потрапляє в стан графа (кроки плану, висновки кроків, рішення replanner, відповідь), проходить normalize().
violations() — ті самі правила для перевірок у demo.py і тестах; вони ловлять і пошкодження, які могла б внести сама
нормалізація (дата, що злиплася з текстом, зіпсований чи подвійний ID, ID не від тієї назви).
"""

import re

from tools import HOTELS

MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
          "листопада", "грудня"]
_M = "|".join(MONTHS)
_ISO = r"\d{4}-\d{2}-\d{2}"
_DAY = r"(?<![\w\-.])(\d{1,2})"  # день — окреме число, а не кінець ID («LVI-001») чи дробу
_DASH = r"\s*(?:–|—|-|по|до)\s*"
_YEAR = r"(\d{4})(?:\s+року\b|\s*р\.)?"
_ID = r"[A-Z]{3}-\d{3}"
# Число з розділювачами розрядів перед «грн» («7 120 грн», «7,120 грн»); не частина дати чи часу
SPACED_SUM = r"(?<![\d\-.:,])\d{1,3}(?:[\s  ,]\d{3})+(?=\s*грн)"
LATIN_CITIES = {"Kyiv": "Київ", "Kiev": "Київ", "Lviv": "Львів", "Odesa": "Одеса", "Odessa": "Одеса"}
# Назви готелів містять латинські назви міст («Lviv Green Garden») — їх не чіпаємо
_HOTEL_NAMES = re.compile("(" + "|".join(re.escape(h["name"]) for h in HOTELS.values()) + ")")

RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b\d{{1,2}}\s+({_M})|\b\d{{1,2}}\.\d{{2}}\.\d{{4}}\b"), "дата не у форматі YYYY-MM-DD"),
    (re.compile(rf"{_ISO}(?: – {_ISO})?\s*[—\-]\s*{_ISO}|{_ISO}–{_ISO}|з {_ISO} (?:до|по) {_ISO}"),
     "період не у форматі «YYYY-MM-DD – YYYY-MM-DD»"),
    (re.compile(SPACED_SUM), "сума з розділювачем розрядів"),
    (re.compile(r"\d,\d"), "десятковий дріб з комою"),
    (re.compile(r"\*\*|__|^#", re.MULTILINE), "Markdown"),
    (re.compile(rf"{_ISO}(?=[^\W\d_])"), "дата злиплася з текстом"),
    (re.compile(r"\b[A-Z]{3}-\d{4,}"), "пошкоджений ID готелю"),
    (re.compile(rf"\b({_ID})\W{{1,3}}\1\b"), "ID готелю повторено"),
    (re.compile(rf"\b{_ID}\s+{_ID}\s+{_HOTEL_NAMES.pattern}"), "два ID перед назвою готелю"),
]


def _iso(day: str, month: str, year: str) -> str:
    return f"{year}-{MONTHS.index(month) + 1:02d}-{int(day):02d}"


def violations(text: str) -> list[str]:
    """Порушення єдиного формату в тексті (порожній список — формат дотримано)."""
    found = [f"{msg}: «{m.group(0)}»" for rx, msg in RULES if (m := rx.search(text))]
    for hid, h in HOTELS.items():
        name = re.escape(h["name"])
        if len(re.findall(name, text)) != len(re.findall(rf"\b{hid} {name}", text)):
            found.append(f"готель не у форматі «ID Назва»: {h['name']}")
        if m := re.search(rf"\b(?!{hid}\b){_ID}\s+{name}", text):
            found.append(f"ID не відповідає назві: «{m.group(0)}»")
    if m := re.search(rf"\b({'|'.join(LATIN_CITIES)})\b", _HOTEL_NAMES.sub("", text)):
        found.append(f"місто латиницею: {m.group(0)}")
    return found


def _year_for(text: str, pos: int, month: str) -> str | None:
    """Рік для дати без року: з ISO-дати того ж місяця, інакше з найближчої попередньої, інакше з першої."""
    dates = list(re.finditer(r"\b(\d{4})-(\d{2})-\d{2}\b", text))
    same_month = [d for d in dates if int(d[2]) == MONTHS.index(month) + 1]
    before = [d for d in dates if d.start() < pos]
    pick = same_month or before[-1:] or dates[:1]
    return pick[0][1] if pick else None


def normalize(text: str) -> str:
    """Привести текст LLM до єдиного формату: дати, періоди, суми, дроби, готелі «ID Назва», міста, без Markdown."""
    # Markdown: виводиться в термінал як є
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Дати словами й dd.mm.yyyy → YYYY-MM-DD (спершу періоди, потім окремі дати)
    text = re.sub(rf"(?:з\s+)?{_DAY}\s+({_M}){_DASH}(\d{{1,2}})\s+({_M})\s+{_YEAR}",
                  lambda m: f"{_iso(m[1], m[2], m[5])} – {_iso(m[3], m[4], m[5])}", text)
    text = re.sub(rf"(?:з\s+)?{_DAY}{_DASH}(\d{{1,2}})\s+({_M})\s+{_YEAR}",
                  lambda m: f"{_iso(m[1], m[3], m[4])} – {_iso(m[2], m[3], m[4])}", text)
    text = re.sub(rf"{_DAY}\s+({_M})\s+{_YEAR}", lambda m: _iso(m[1], m[2], m[3]), text)
    text = re.sub(rf"{_DAY}\.(\d{{2}})\.(\d{{4}})(?![\d.])", lambda m: f"{m[3]}-{m[2]}-{int(m[1]):02d}", text)

    def no_year_period(m: re.Match) -> str:
        year = _year_for(text, m.start(), m[3])
        return f"{_iso(m[1], m[3], year)} – {_iso(m[2], m[3], year)}" if year else m[0]

    def no_year_date(m: re.Match) -> str:
        year = _year_for(text, m.start(), m[2])
        return _iso(m[1], m[2], year) if year else m[0]

    text = re.sub(rf"(?:з\s+)?{_DAY}{_DASH}(\d{{1,2}})\s+({_M})\b", no_year_period, text)
    text = re.sub(rf"{_DAY}\s+({_M})\b", no_year_date, text)
    # Періоди: «з A по B», «A–B», «A - B» → «A – B»
    text = re.sub(rf"з ({_ISO}) (?:до|по) ({_ISO})", r"\1 – \2", text)
    text = re.sub(rf"({_ISO})\s*[–—-]\s*({_ISO})", r"\1 – \2", text)
    # Суми без розділювачів розрядів, дроби з крапкою
    text = re.sub(SPACED_SUM, lambda m: re.sub(r"\D", "", m[0]), text)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    # Готель — «ID Назва»: прибрати дужки й лапки навколо назви чи ID, а перед назвою поставити саме її ID
    for hid, h in HOTELS.items():
        name = re.escape(h["name"])
        text = re.sub(rf"{name}\s*\(\s*{hid}\s*\)|\(\s*{hid}\s*\)\s*{name}"
                      rf"|\b{hid}\s*[—–\-|:,]?\s*[«\"(]\s*{name}\s*[»\")]|\b{hid}\s*[—–\-|:,]\s*{name}",
                      f"{hid} {h['name']}", text)
        text = re.sub(rf"[«\"]({name})[»\"]", r"\1", text)
        text = re.sub(rf"(?:\b{_ID}\s+)?{name}", f"{hid} {h['name']}", text)
    # Міста українською (поза назвами готелів)
    parts = _HOTEL_NAMES.split(text)
    for i in range(0, len(parts), 2):
        for latin, city in LATIN_CITIES.items():
            parts[i] = re.sub(rf"\b{latin}\b", city, parts[i])
    return "".join(parts)
