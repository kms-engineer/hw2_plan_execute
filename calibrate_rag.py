"""Калібрування порогу search_knowledge для пошуку без topics. Запуск: python calibrate_rag.py

На розміченому наборі запитів (запит → теми, що мають повернутися) рахує точність, повноту й частку запитів,
де повернулися рівно потрібні теми, для сітки порогів. Ранжування — те саме, що в search_knowledge (knowledge.rank),
тож після зміни правил чи ембедингів достатньо перезапустити скрипт і за потреби оновити MIN_RELEVANCE /
RELEVANCE_MARGIN у knowledge.py. test_knowledge.py перевіряє, що поточний поріг не гірший за зафіксований рівень.
"""

from knowledge import MIN_RELEVANCE, RELEVANCE_MARGIN, Hit, rank, select

# (запит, hotel_id, теми, що мають повернутися)
CASES: list[tuple[str, str | None, set[str]]] = [
    ("о котрій заселення", None, {"заселення"}),
    ("час заїзду та виїзду", None, {"заселення"}),
    ("о котрій можна заселитися в готель", "ODS-003", {"заселення"}),
    ("які документи потрібні для заселення", None, {"документи"}),
    ("чи можна з собакою і скільки це коштує", "LVI-002", {"тварини"}),
    ("правила проживання з тваринами", None, {"тварини"}),
    ("о котрій сніданок", "KYV-003", {"харчування"}),
    ("чи є сніданок", "ODS-003", {"харчування"}),
    ("де припаркувати машину", "LVI-001", {"паркінг"}),
    ("чи є парковка біля готелю", "KYV-002", {"паркінг"}),
    ("чи є басейн і до котрої працює", "ODS-001", {"зручності"}),
    ("що робити під час повітряної тривоги", "KYV-001", {"безпека"}),
    ("де найближче укриття", "ODS-002", {"безпека"}),
    ("який депозит при заселенні", "ODS-003", {"депозит"}),
    ("чи повернуть гроші, якщо скасувати бронювання", None, {"скасування"}),
    ("як доїхати потягом до Львова", None, {"транспорт"}),
    ("чи є трансфер з вокзалу", "KYV-002", {"трансфер"}),
    ("скільки туристичний збір", None, {"збір"}),
    ("чи можна курити в номері", "LVI-003", {"розпорядок"}),
    ("чи буде світло під час відключень", None, {"відключення"}),
    ("чи дадуть дитяче ліжечко", None, {"діти"}),
    ("як оплатити проживання карткою", None, {"оплата"}),
    ("що подивитися в Одесі", None, {"місто"}),
    # питання про кілька тем
    ("О котрій заселення й виселення і які документи потрібні для заселення?", None, {"заселення", "документи"}),
    ("Загальні правила мережі: час заселення та виселення, які документи потрібні для заселення", None,
     {"заселення", "документи"}),
    ("правила з собакою і де паркуватися", "LVI-002", {"тварини", "паркінг"}),
    ("депозит і як оплатити", None, {"депозит", "оплата"}),
]


def rankings() -> list[tuple[list[Hit], set[str]]]:
    """Ранжування кожного запиту набору (один виклик ембедингів на запит)."""
    return [(rank(query, hotel_id), want) for query, hotel_id, want in CASES]


def evaluate(ranked: list[tuple[list[Hit], set[str]]], min_relevance: float = MIN_RELEVANCE,
             margin: float = RELEVANCE_MARGIN) -> tuple[float, float, int]:
    """Точність, повнота й кількість запитів, де select() повернув рівно потрібні теми."""
    tp = fp = fn = exact = 0
    for hits, want in ranked:
        got = {str(meta["topic"]) for _, _, meta in select(hits, min_relevance, margin)}
        tp, fp, fn, exact = tp + len(got & want), fp + len(got - want), fn + len(want - got), exact + (got == want)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), exact


if __name__ == "__main__":
    ranked = rankings()
    for (query, hotel_id, want), (hits, _) in zip(CASES, ranked):
        top = ", ".join(f"{meta['topic']} {score:.2f}" for score, _, meta in hits[:3])
        print(f"{query[:55]:55} {hotel_id or '':7} очікується: {', '.join(sorted(want)):22} топ: {top}")
    print("\nмінімум  відступ  точність  повнота  рівно потрібні теми")
    for min_relevance in (0.25, 0.3, 0.35, 0.4):
        for margin in (0.05, 0.1, 0.15, 0.2):
            precision, recall, exact = evaluate(ranked, min_relevance, margin)
            mark = "  ← поточний" if (min_relevance, margin) == (MIN_RELEVANCE, RELEVANCE_MARGIN) else ""
            print(f"{min_relevance:7}  {margin:7}  {precision:8.2f}  {recall:7.2f}  {exact:3}/{len(CASES)}{mark}")
