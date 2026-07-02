"""
add_math_data.py — דאטה שמלמדת את המודל *להבין* בקשת חישוב, לא לשנן תשובות.

גישת "מודל מבין → כלי מחשב" (tool use):
  התשובה של המודל לבקשת חישוב אינה המספר, אלא **קריאה מובנית** לכלי:
      User: 7 כפול 8   Model: <calc> *
  בזמן ריצה: המודל פולט <calc> ואת הפעולה שהבין; כלי דטרמיניסטי מחלץ את
  שני המספרים מהקלט (תמיד מדויק — הם שם מילולית) ומחשב לכל גודל.

מה הרשת לומדת כאן:
  1. *מתי* זו בקשת חישוב (כוונה).
  2. *איזו* פעולה — גם כשהיא כתובה במילים ("כפול", "ההפרש בין", "פי").
המספרים עצמם לא נלמדים: המודל נשען על טוקן-האופרטור/מילת-הפעולה, ולכן
הדפוס מכליל לכל מספר (גם ל-<UNK> וגם למספרים ענקיים שלא נראו).

לכן דוגמים מספרים מגוונים בכוונה (קטנים/בינוניים/גדולים) — כדי שהמספר
*לא* יהיה אות מנבא, אלא רק הפעולה.
"""

from __future__ import annotations

import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "math_training_data.json")

CALC = "<calc>"
random.seed(42)

# לכל פעולה: הסמל (מה שהמודל פולט) ורשימת תבניות ניסוח לקלט המשתמש.
# {a},{b} מוחלפים במספרים. הצורות כוללות סמל מפורש וגם ניסוח מילולי.
#
# שני אילוצים על התבניות (בגלל context_len וסדר החילוץ):
#   1. סדר האופרנדים תמיד "a ואז b" — הכלי מחלץ את שני המספרים הראשונים
#      לפי סדר הופעתם, אז ניסוח שמהפך סדר ("כמה פעמים b נכנס ב a") אסור.
#   2. מילת-הפעולה חייבת להיכנס לחלון-ההקשר בזמן שהמודל פולט את הפעולה;
#      context_len=8 מאפשר גם ניסוחים שמילת-הפעולה בהם מוקדמת ("הסכום של").
OPERATIONS = {
    "+": [
        "{a} + {b}", "כמה זה {a} + {b}",
        "{a} ועוד {b}", "כמה זה {a} ועוד {b}", "{a} פלוס {b}",
        "הסכום של {a} ו {b}", "כמה זה {a} plus {b}",
        "{a} plus {b}", "what is {a} + {b}",
    ],
    "-": [
        "{a} - {b}", "כמה זה {a} - {b}",
        "{a} פחות {b}", "כמה זה {a} פחות {b}", "{a} מינוס {b}",
        "ההפרש בין {a} ל {b}", "כמה זה {a} minus {b}",
        "{a} minus {b}", "what is {a} - {b}",
    ],
    "*": [
        "{a} * {b}", "כמה זה {a} * {b}",
        "{a} כפול {b}", "כמה זה {a} כפול {b}", "{a} פעמים {b}",
        "{a} פי {b}", "המכפלה של {a} ו {b}",
        "{a} times {b}", "what is {a} * {b}",
    ],
    "/": [
        "{a} / {b}", "כמה זה {a} / {b}",
        "{a} חלקי {b}", "כמה זה {a} חלקי {b}", "{a} לחלק ל {b}",
        "{a} מחולק ב {b}", "{a} divided by {b}",
        "what is {a} / {b}",
    ],
}

# כמה זוגות מספרים אקראיים לכל תבנית — מגוון גדלים כדי שהמספר לא ינבא.
PAIRS_PER_TEMPLATE = 12


def sample_number() -> int:
    """מספר מגוון: לרוב קטן, לפעמים בינוני/גדול — כדי ללמד אי-תלות בגודל."""
    bucket = random.random()
    if bucket < 0.55:
        return random.randint(0, 20)
    if bucket < 0.85:
        return random.randint(21, 199)
    return random.randint(200, 9999)


def build_articles() -> list[dict]:
    articles: list[dict] = []
    idx = 0
    for op, templates in OPERATIONS.items():
        for template in templates:
            for _ in range(PAIRS_PER_TEMPLATE):
                a, b = sample_number(), sample_number()
                # לחיסור/חילוק נעדיף a>=b כדי להימנע משליליים/שברים מוזרים בדוגמה
                if op in "-/" and b > a:
                    a, b = b, a
                if op == "/" and b == 0:
                    b = 1
                question = template.format(a=a, b=b)
                text = f"User: {question} Model: {CALC} {op}"
                articles.append({
                    "id": f"calc_{op}_{idx}",
                    "text": text,
                    "source": "math_generator",
                    "topic": "arithmetic_tool",
                })
                idx += 1
    random.shuffle(articles)
    return articles


def main() -> None:
    articles = build_articles()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"articles": articles}, f, ensure_ascii=False, indent=2)
    by_op = {}
    for a in articles:
        op = a["id"].split("_")[1]
        by_op[op] = by_op.get(op, 0) + 1
    print(f"[Math] נכתבו {len(articles)} דוגמאות קריאה-לכלי → {os.path.basename(OUT_PATH)}")
    print(f"[Math] לפי פעולה: {by_op}")
    print(f"[Math] דוגמא: {articles[0]['text']!r}")


if __name__ == "__main__":
    main()
