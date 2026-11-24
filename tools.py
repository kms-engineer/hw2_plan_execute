"""Завдання 1. Інструменти (tools) агента бронювання готелів з Pydantic v2 валідацією (перевикористано з ДЗ1).

Домен: пошук і бронювання готелів в Україні (Київ, Львів, Одеса).
book_hotel — ризикова незворотна дія: граф виконує її лише після підтвердження людини (hitl.py).
"""

import hashlib
import math
import re
from datetime import date
from typing import TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


# ── Mock-база готелів ────────────────────────────────────────────
# price — ціна номера за ніч (грн); max_guests — місткість номера; rooms_left — вільні номери;
# breakfast — ціна сніданку за особу/день (None — сніданок не надається).
class Hotel(TypedDict):
    name: str
    city: str
    stars: int
    rating: float
    price: int
    max_guests: int
    rooms_left: int
    breakfast: int | None
    address: str
    center_km: float
    amenities: list[str]
    free_cancellation: bool
    pets: bool


HOTELS: dict[str, Hotel] = {
    "KYV-001": {"name": "Dnipro View Hotel", "city": "Київ", "stars": 4, "rating": 8.9, "price": 3200,
                "max_guests": 2, "rooms_left": 5, "breakfast": 350, "address": "вул. Хрещатик, 12",
                "center_km": 0.5, "amenities": ["Wi-Fi", "спортзал", "паркінг"],
                "free_cancellation": True, "pets": False},
    "KYV-002": {"name": "Podil Boutique", "city": "Київ", "stars": 4, "rating": 9.3, "price": 3800,
                "max_guests": 2, "rooms_left": 2, "breakfast": 300, "address": "вул. Сагайдачного, 25",
                "center_km": 1.8, "amenities": ["Wi-Fi", "бар", "трансфер"],
                "free_cancellation": False, "pets": True},
    "KYV-003": {"name": "Kyiv Budget Inn", "city": "Київ", "stars": 2, "rating": 7.6, "price": 1300,
                "max_guests": 3, "rooms_left": 10, "breakfast": None, "address": "просп. Перемоги, 80",
                "center_km": 6.0, "amenities": ["Wi-Fi"], "free_cancellation": True, "pets": False},
    "LVI-001": {"name": "Rynok Square Hotel", "city": "Львів", "stars": 4, "rating": 9.1, "price": 2900,
                "max_guests": 2, "rooms_left": 3, "breakfast": 280, "address": "пл. Ринок, 7",
                "center_km": 0.1, "amenities": ["Wi-Fi", "ресторан", "кав'ярня"],
                "free_cancellation": True, "pets": False},
    "LVI-002": {"name": "Lviv Green Garden", "city": "Львів", "stars": 3, "rating": 8.4, "price": 1900,
                "max_guests": 3, "rooms_left": 6, "breakfast": 220, "address": "вул. Личаківська, 40",
                "center_km": 1.5, "amenities": ["Wi-Fi", "сад", "паркінг"],
                "free_cancellation": True, "pets": True},
    "LVI-003": {"name": "Opera Grand", "city": "Львів", "stars": 5, "rating": 9.5, "price": 5200,
                "max_guests": 2, "rooms_left": 0, "breakfast": 450, "address": "просп. Свободи, 28",
                "center_km": 0.4, "amenities": ["Wi-Fi", "SPA", "басейн"],
                "free_cancellation": False, "pets": False},
    "ODS-001": {"name": "Arcadia Beach Resort", "city": "Одеса", "stars": 4, "rating": 8.7, "price": 3500,
                "max_guests": 3, "rooms_left": 4, "breakfast": 400, "address": "Генуезька вул., 24",
                "center_km": 6.5, "amenities": ["пляж", "басейн", "Wi-Fi"],
                "free_cancellation": True, "pets": False},
    "ODS-002": {"name": "Deribasivska Residence", "city": "Одеса", "stars": 3, "rating": 8.1, "price": 2100,
                "max_guests": 2, "rooms_left": 8, "breakfast": 250, "address": "вул. Дерибасівська, 15",
                "center_km": 0.2, "amenities": ["Wi-Fi", "кухня"], "free_cancellation": True, "pets": True},
    "ODS-003": {"name": "Black Sea Hostel", "city": "Одеса", "stars": 1, "rating": 7.2, "price": 700,
                "max_guests": 4, "rooms_left": 12, "breakfast": None, "address": "вул. Пушкінська, 50",
                "center_km": 1.0, "amenities": ["Wi-Fi", "спільна кухня"],
                "free_cancellation": False, "pets": False},
}

# Назви міст, які розуміє пошук (укр/англ) → канонічна назва (як у HOTELS і в усіх виходах tools)
CITY_ALIASES = {"kyiv": "Київ", "kiev": "Київ", "київ": "Київ", "києві": "Київ",
                "lviv": "Львів", "львів": "Львів", "львові": "Львів",
                "odesa": "Одеса", "odessa": "Одеса", "одеса": "Одеса", "одесі": "Одеса"}

TOURIST_TAX = 50   # туристичний збір, грн за особу за ніч (mock)
MAX_NIGHTS = 30
MAX_GUESTS = 8
MAX_RATING = 10        # шкала рейтингу гостей: 0–MAX_RATING
FREE_CANCEL_HOURS = 48  # безкоштовне скасування — не пізніше ніж за стільки годин до заїзду
BOOKINGS: dict[str, dict] = {}  # підтверджені бронювання (in-memory)


def rooms_needed(hotel: Hotel, guests: int) -> int:
    return math.ceil(guests / hotel["max_guests"])


def has_rooms(hotel_id: str, guests: int) -> bool:
    """Чи вистачить вільних номерів — спільна перевірка для book_hotel і вузла підтвердження (hitl.py)."""
    h = HOTELS[hotel_id]
    return h["rooms_left"] >= rooms_needed(h, guests)


def cancellation(h: Hotel) -> str:
    return f"безкоштовне до {FREE_CANCEL_HOURS} год до заїзду" if h["free_cancellation"] else "без повернення коштів"


def doc_vars(**values):
    """Підставляє константи коду в docstring tool: опис для LLM не розходиться з реальними лімітами.

    Ставиться під @tool — той читає docstring уже з підставленими значеннями."""
    def apply(fn):
        fn.__doc__ = fn.__doc__.format(**values)
        return fn
    return apply


DOC_LIMITS = {"max_nights": MAX_NIGHTS, "max_guests": MAX_GUESTS, "tax": TOURIST_TAX, "cancel_hours": FREE_CANCEL_HOURS}


def check_hotel_id(v: str) -> str:
    code = v.strip().upper()
    if code not in HOTELS:
        raise ValueError(f'Готель "{v}" не знайдено. ID має формат KYV-001 — візьміть його з результатів search_hotels')
    return code


# ── Спільна схема дат і гостей (успадковують інші схеми) ─────────
class StayInput(BaseModel):
    """Дати проживання і кількість гостей."""

    model_config = ConfigDict(str_strip_whitespace=True)

    check_in: date = Field(description="Дата заїзду у форматі YYYY-MM-DD, не раніше сьогодні")
    check_out: date = Field(
        description=f"Дата виїзду у форматі YYYY-MM-DD, пізніше за check_in (макс. {MAX_NIGHTS} ночей)")
    guests: int = Field(default=2, description=f"Кількість гостей (1–{MAX_GUESTS})")

    @field_validator("check_in")
    @classmethod
    def check_in_not_past(cls, v: date) -> date:
        if v < date.today():
            raise ValueError(f"Дата заїзду {v} уже минула (сьогодні {date.today()})")
        return v

    @field_validator("check_out")
    @classmethod
    def check_out_after_check_in(cls, v: date, info: ValidationInfo) -> date:
        check_in = info.data.get("check_in")  # відсутня, якщо check_in не пройшла валідацію
        if check_in and v <= check_in:
            raise ValueError("Дата виїзду має бути пізніше за дату заїзду")
        if check_in and (v - check_in).days > MAX_NIGHTS:
            raise ValueError(f"Максимальний строк проживання — {MAX_NIGHTS} ночей")
        return v

    @field_validator("guests")
    @classmethod
    def guests_range(cls, v: int) -> int:
        if not 1 <= v <= MAX_GUESTS:
            raise ValueError(f"Кількість гостей має бути від 1 до {MAX_GUESTS}")
        return v


# ── Tool 1: пошук готелів ────────────────────────────────────────
class SearchHotelsInput(StayInput):
    """Параметри пошуку готелів."""

    city: str = Field(description='Місто: "Київ", "Львів" або "Одеса" (можна англійською: "Kyiv", "Lviv", "Odesa")')
    max_price_per_night: float | None = Field(default=None, description="Макс. ціна номера за ніч, грн (необов'язково)")
    min_rating: float | None = Field(
        default=None, description=f"Мінімальний рейтинг гостей від 0 до {MAX_RATING} (необов'язково)")

    @field_validator("city")
    @classmethod
    def city_supported(cls, v: str) -> str:
        city = CITY_ALIASES.get(v.lower())
        if not city:
            raise ValueError(f'Місто "{v}" не підтримується. Доступні: Київ, Львів, Одеса')
        return city

    @field_validator("max_price_per_night")
    @classmethod
    def price_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("Максимальна ціна має бути більшою за 0")
        return v

    @field_validator("min_rating")
    @classmethod
    def rating_range(cls, v: float | None) -> float | None:
        if v is not None and not 0 <= v <= MAX_RATING:
            raise ValueError(f"Рейтинг має бути від 0 до {MAX_RATING}")
        return v


@tool(args_schema=SearchHotelsInput)
@doc_vars(**DOC_LIMITS)
def search_hotels(check_in: date, check_out: date, city: str, guests: int = 2,
                  max_price_per_night: float | None = None, min_rating: float | None = None) -> str:
    """Шукає готелі з вільними номерами в Києві, Львові або Одесі на задані дати.

    Коли використовувати:
    - перший крок, якщо користувач шукає житло або хоче бронювати, але hotel_id ще невідомий;
    - щоб порівняти готелі одного міста за ціною та рейтингом.

    Коли НЕ використовувати:
    - hotel_id відомий, а потрібні умови (скасування, тварини, сніданок) → get_hotel_details;
    - потрібна повна сума зі сніданком і збором → calculate_total_price;
    - місто не Київ/Львів/Одеса → не викликай; скажи, що це місто не підтримується (лише Київ, Львів, Одеса).

    Обмеження:
    - дати YYYY-MM-DD, без часу й часового поясу («сьогодні» — локальна дата сервера);
      заїзд не в минулому, до {max_nights} ночей, 1–{max_guests} гостей;
    - ціни в грн за номер за ніч, без сніданку й збору; «за N ноч.» = ціна × ночі × кількість номерів;
    - кілька номерів, якщо гостей більше за місткість; готелі без вільних номерів не показано; сортування за рейтингом.

    Помилки:
    - «не підтримується» (місто) — НЕ retryable: запропонуй Київ, Львів або Одесу;
    - помилки дат («уже минула», «пізніше за дату заїзду», «{max_nights} ночей») — retryable після виправлення дат;
    - «немає готелів за цими фільтрами» — не помилка: повідом і запропонуй послабити фільтри.
    """
    nights = (check_out - check_in).days
    header = f"Готелі — {city}, {check_in} – {check_out} ({nights} ноч.), гостей: {guests}"
    lines = []
    for hid, h in sorted(HOTELS.items(), key=lambda kv: -kv[1]["rating"]):
        rooms = rooms_needed(h, guests)
        if (h["city"] != city or h["rooms_left"] < rooms
                or (max_price_per_night and h["price"] > max_price_per_night)
                or (min_rating and h["rating"] < min_rating)):
            continue
        lines.append(f"- {hid} {h['name']} | {h['stars']}★ | рейтинг {h['rating']} | {h['price']} грн/ніч за номер | "
                     f"номерів потрібно: {rooms} | за {nights} ноч.: {h['price'] * nights * rooms} грн | "
                     f"{h['center_km']} км від центру")
    if not lines:
        return f"{header}: немає готелів за цими фільтрами."
    return f"{header} (ціни без сніданку й туристичного збору):\n" + "\n".join(lines)


# ── Tool 2: деталі готелю ────────────────────────────────────────
class HotelDetailsInput(BaseModel):
    """Параметри запиту деталей готелю."""

    hotel_id: str = Field(description='ID готелю з результатів search_hotels, напр. "KYV-001"')

    @field_validator("hotel_id")
    @classmethod
    def hotel_exists(cls, v: str) -> str:
        return check_hotel_id(v)


@tool(args_schema=HotelDetailsInput)
@doc_vars(**DOC_LIMITS)
def get_hotel_details(hotel_id: str) -> str:
    """Повертає факти про один готель: адресу, зручності, сніданок, тварин, тариф скасування, місткість номера,
    а також теми, де в готелю свої правила проживання («Особливі правила готелю»).

    Коли використовувати:
    - користувач питає про конкретний готель (відомий hotel_id);
    - треба перевірити умову, якої немає в search_hotels: скасування, тварини, ціна сніданку.

    Коли НЕ використовувати:
    - щоб знайти готелі чи перевірити наявність на дати → search_hotels (деталі від дат не залежать);
    - щоб порахувати вартість → calculate_total_price;
    - умови з правил готелю (вага й доплата за тварин, паркінг, час заїзду, депозит) → search_knowledge з hotel_id
      і topics із «Особливих правил готелю»; тут лише чи дозволено, а не на яких умовах;
    - повторно для того самого hotel_id — дані статичні.

    Обмеження:
    - hotel_id у форматі KYV-001 / LVI-002 / ODS-003 (регістр і пробіли не важливі);
    - сніданок — грн за особу за день; безкоштовне скасування — до {cancel_hours} год до заїзду;
    - «вільних номерів» — загальна кількість, не на конкретні дати.

    Помилки:
    - «не знайдено» (готель) — НЕ retryable з тим самим ID: не вгадуй ID, візьми його з search_hotels
      або скажи користувачу, що такого готелю немає.
    """
    h = HOTELS[hotel_id]
    breakfast = f"{h['breakfast']} грн/особа/день" if h["breakfast"] else "не надається"
    return (
        f"{hotel_id} {h['name']} ({h['stars']}★, рейтинг {h['rating']}), {h['city']}, {h['address']}, "
        f"{h['center_km']} км від центру. Зручності: {', '.join(h['amenities'])}. Сніданок: {breakfast}. "
        f"Тварини: {'дозволені' if h['pets'] else 'заборонені (окрім собак-поводирів)'}. "
        f"Скасування: {cancellation(h)}. "
        f"Номер до {h['max_guests']} гостей, вільних номерів: {h['rooms_left']}."
        f"{policy_topics(hotel_id)}"
    )


def policy_topics(hotel_id: str) -> str:
    """Рядок картки з темами, де в готелю свої правила (самі правила — у search_knowledge)."""
    from knowledge import hotel_policy_topics  # тут, а не вгорі: knowledge імпортує tools

    topics = hotel_policy_topics(hotel_id)
    return f" Особливі правила готелю (search_knowledge з hotel_id): {', '.join(topics)}." if topics else ""


# ── Tool 3: розрахунок вартості ──────────────────────────────────
class PriceInput(StayInput):
    """Параметри розрахунку вартості проживання."""

    hotel_id: str = Field(description='ID готелю, напр. "KYV-001"')
    breakfast: bool = Field(default=False, description="True — додати сніданок для всіх гостей щоранку (ночі × гості)")

    @field_validator("hotel_id")
    @classmethod
    def hotel_exists(cls, v: str) -> str:
        return check_hotel_id(v)


def price_breakdown(hotel_id: str, check_in: date, check_out: date, guests: int, breakfast: bool) -> dict:
    """Спільний розрахунок вартості для calculate_total_price, book_hotel і запиту підтвердження (hitl.py)."""
    h = HOTELS[hotel_id]
    nights = (check_out - check_in).days
    rooms = rooms_needed(h, guests)
    room_cost = h["price"] * nights * rooms
    breakfast_cost = h["breakfast"] * guests * nights if breakfast and h["breakfast"] else 0
    tax = TOURIST_TAX * guests * nights
    return {"nights": nights, "rooms": rooms, "room_cost": room_cost, "breakfast_cost": breakfast_cost,
            "tax": tax, "total": room_cost + breakfast_cost + tax,
            "breakfast_note": " (сніданок у цьому готелі не надається)" if breakfast and not h["breakfast"] else ""}


@tool(args_schema=PriceInput)
@doc_vars(**DOC_LIMITS)
def calculate_total_price(hotel_id: str, check_in: date, check_out: date, guests: int = 2,
                          breakfast: bool = False) -> str:
    """Рахує вартість проживання в одному готелі: номери + сніданки + туристичний збір.

    Коли використовувати:
    - користувач питає «скільки коштуватиме», «яка повна ціна»;
    - перед бронюванням, щоб назвати підсумкову суму;
    - щоб порівняти реальні суми готелів, коли групі потрібно кілька номерів.

    Коли НЕ використовувати:
    - для пошуку готелів → search_hotels (ціна за ніч уже є там);
    - після book_hotel — сума вже є в підтвердженні бронювання.

    Обмеження:
    - суми в грн; номери = ціна за ніч × ночі × кількість номерів;
    - сніданок = ціна × гості × ночі (лише якщо breakfast=True і готель його надає); збір {tax} грн/особа/ніч;
    - НЕ входять додаткові послуги й доплати (тварини, ранній заїзд / пізній виїзд, паркінг, трансфер тощо)
      і депозит — їх розмір дає search_knowledge з hotel_id (правила готелю, а де їх немає — загальні);
      називай їх окремо від суми;
    - наявність номерів НЕ перевіряє; дати й гості — як у search_hotels (YYYY-MM-DD, до {max_nights} ночей,
      1–{max_guests} гостей).

    Помилки:
    - «не знайдено» (готель) — НЕ retryable з тим самим ID (див. get_hotel_details);
    - помилки дат — retryable після виправлення;
    - «сніданок у цьому готелі не надається» — не помилка: сума без сніданку, повідом користувача.
    """
    p = price_breakdown(hotel_id, check_in, check_out, guests, breakfast)
    return (
        f"{hotel_id} {HOTELS[hotel_id]['name']}, {check_in} – {check_out} ({p['nights']} ноч.), гостей: {guests}: "
        f"номери ({p['rooms']} шт.) {p['room_cost']} грн + сніданки {p['breakfast_cost']} грн{p['breakfast_note']} "
        f"+ туристичний збір {p['tax']} грн = РАЗОМ {p['total']} грн"
    )


def booking_summary(hotel_id: str, check_in: date, check_out: date, guests: int, breakfast: bool,
                    guest_name: str, email: str) -> str:
    """Опис бронювання одним форматом для запиту підтвердження (hitl.py) і відповіді book_hotel."""
    h = HOTELS[hotel_id]
    p = price_breakdown(hotel_id, check_in, check_out, guests, breakfast)
    return (f"{hotel_id} {h['name']} ({h['city']}), {check_in} – {check_out} ({p['nights']} ноч.), "
            f"гостей: {guests}, номерів: {p['rooms']}, сніданок: {'так' if breakfast and h['breakfast'] else 'ні'}, "
            f"гість: {guest_name} <{email}>, до сплати {p['total']} грн, скасування: {cancellation(h)}")


# ── Tool 4: бронювання ───────────────────────────────────────────
class BookHotelInput(PriceInput):
    """Параметри бронювання (дати, гості, готель, сніданок + дані гостя)."""

    guest_name: str = Field(description="Ім'я та прізвище гостя, на кого бронювати (напр. \"Іван Петренко\")")
    email: str = Field(description="Email гостя для підтвердження бронювання")

    @field_validator("guest_name")
    @classmethod
    def name_valid(cls, v: str) -> str:
        if len(v.split()) < 2 or any(ch.isdigit() for ch in v):
            raise ValueError("Вкажіть ім'я та прізвище гостя (без цифр), напр. \"Іван Петренко\"")
        return v

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v):
            raise ValueError(f'Некоректний email "{v}". Очікується формат name@example.com')
        return v.lower()


@tool(args_schema=BookHotelInput)
def book_hotel(hotel_id: str, check_in: date, check_out: date, guest_name: str, email: str,
               guests: int = 2, breakfast: bool = False) -> str:
    """Бронює номер(и) в готелі й повертає код підтвердження BK-XXXXXX і суму до сплати (РИЗИКОВА ДІЯ).

    Коли використовувати:
    - користувач явно попросив забронювати і назвав ім'я, прізвище та email гостя;
    - останнім кроком, після вибору готелю (search_hotels) і, за потреби, перевірки умов і ціни.

    Коли НЕ використовувати:
    - користувач лише питає про готелі, ціни чи умови;
    - бракує імені або email — не вигадуй їх, попроси користувача;
    - скасування чи зміна бронювання — такого tool немає: правила дає search_knowledge, оформлює рецепція.

    Обмеження:
    - guest_name — ім'я та прізвище без цифр; email у форматі name@example.com;
    - дати й гості — як у search_hotels; сума рахується так само, як у calculate_total_price;
    - підтвердження людини (approve / reject / edit) граф запитує автоматично перед виконанням — не питай його сам;
    - той самий запит повертає той самий код (дубль не створюється), будь-яка зміна аргументів — новий код.

    Помилки:
    - «Оператор ВІДХИЛИВ» — НЕ retryable: не повторюй виклик, повідом користувачу причину;
    - помилка імені чи email — retryable, лише якщо правильні дані є в запиті користувача, інакше спитай його;
    - «Бронювання неможливе» (немає номерів) — НЕ retryable для цього готелю: запропонуй інший з search_hotels;
    - помилки дат — retryable після виправлення.
    """
    h = HOTELS[hotel_id]
    p = price_breakdown(hotel_id, check_in, check_out, guests, breakfast)
    if not has_rooms(hotel_id, guests):
        return (f"Бронювання неможливе: у {hotel_id} {h['name']} недостатньо вільних номерів "
                f"(вільних: {h['rooms_left']}, потрібно: {p['rooms']}).")
    # Детермінований код: однакові дані → той самий код (повторний виклик не створює дубль)
    key = f"{hotel_id}{check_in}{check_out}{guests}{breakfast}{guest_name}{email}"
    code = "BK-" + hashlib.md5(key.encode()).hexdigest()[:6].upper()
    # Обмеження: mock-бронювання не зменшує rooms_left — доступність статична, щоб тести були відтворюваними
    BOOKINGS[code] = {"hotel_id": hotel_id, "check_in": str(check_in), "check_out": str(check_out),
                      "guests": guests, "guest_name": guest_name, "email": email, "total": p["total"]}
    summary = booking_summary(hotel_id, check_in, check_out, guests, breakfast, guest_name, email)
    return f"Бронювання підтверджено! Код: {code}. {summary}. Підтвердження надіслано на {email}."


TOOLS = [search_hotels, get_hotel_details, calculate_total_price, book_hotel]
