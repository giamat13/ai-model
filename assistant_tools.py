"""
assistant_tools.py - כלים חיצוניים למודל הזעיר.

המודל עצמו קטן ולכן הוא לא אמור לנחש בכוח. הקובץ הזה מוסיף:
  * מחולל קוד Python חכם לפקודות כתיבת קוד.
  * חיפוש קצר בויקיפדיה לשאלות ידע.
  * החלטת "אני לא יודע" כאשר אין מספיק מידע.

הערה: המחשבון האלגוריתמי (ast + eval) הוסר בכוונה. חשבון אינו עוד "כלי"
חיצוני — המודל הנוירוני עצמו לומד לחשב מתוך דוגמאות אימון (ראה
training_data.json, ראה add_math_data.py). שאלות חשבון נופלות ל-generate() של המודל.
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import json
from dataclasses import dataclass
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES_DIR = os.path.join(_HERE, "sources")
SOURCE_CACHE: list["SourceDocument"] | None = None


@dataclass
class SourceDocument:
    path: str
    name: str
    text: str


UNKNOWN_HE = "אני לא יודע."
UNKNOWN_EN = "I don't know."
USER_AGENT = "mini-bilingual-ai-assistant/1.0"


def has_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def unknown_answer(text: str) -> str:
    return UNKNOWN_HE if has_hebrew(text) else UNKNOWN_EN


# ════════════════════════════════════════════════════════════════
#  GREETINGS & SMALL TALK — לפני חיפוש נושא/ויקיפדיה
# ════════════════════════════════════════════════════════════════

GREETING_WORDS_HE = {"שלום", "הי", "היי", "אהלן", "יאסו"}
GREETING_WORDS_EN = {"hi", "hello", "hey", "yo", "sup"}
GREETING_PHRASES_HE = {
    "שלום", "היי", "הי", "אהלן", "בוקר טוב", "ערב טוב", "לילה טוב",
    "מה נשמע", "מה קורה", "מה המצב", "מה שלומך", "שלום מה שלומך",
    "שלום מה נשמע",
}
GREETING_PHRASES_EN = {
    "hi", "hello", "hey", "yo", "sup", "good morning", "good evening",
    "good afternoon", "whats up", "what's up", "how are you", "hows it going",
}

GREETING_REPLIES_HE = ["שלום! איך אפשר לעזור?", "היי! מה שלומך?", "שלום, אני כאן בשבילך."]
GREETING_REPLIES_EN = ["Hello! How can I help?", "Hi there! What can I do for you?", "Hey! I'm here to help."]


def is_greeting(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if norm in GREETING_PHRASES_HE or norm in GREETING_PHRASES_EN:
        return True
    words = norm.split()
    first = words[0]
    return len(words) <= 3 and (first in GREETING_WORDS_HE or first in GREETING_WORDS_EN)


def greeting_answer(text: str) -> str:
    import random
    replies = GREETING_REPLIES_HE if has_hebrew(text) else GREETING_REPLIES_EN
    return random.choice(replies)


# ════════════════════════════════════════════════════════════════
#  SELF-AWARENESS — שאלות זהות ("מי אתה")
# ════════════════════════════════════════════════════════════════

IDENTITY_PATTERN_HE = re.compile(
    r"מי את[הן]|מה את[הן]\b|את[הן] בינה מלאכותית|את[הן] רובוט|את[הן] בוט|את[הן] אנושי"
)
IDENTITY_PATTERN_EN = re.compile(
    r"who are you|what are you|are you an? (ai|bot|robot|human)", re.IGNORECASE
)

IDENTITY_ANSWER_HE = (
    "אני עוזר בינה מלאכותית קטן שנבנה עם NumPy. "
    "אני יודע לחשב, לכתוב קוד, לחפש מידע ולשוחח איתך — אני לא אדם ואין לי זיכרונות אישיים משלי."
)
IDENTITY_ANSWER_EN = (
    "I'm a small AI assistant built with NumPy. "
    "I can calculate, write code, look things up, and chat — I'm not a person and I don't have personal memories of my own."
)


def is_identity_question(text: str) -> bool:
    return bool(IDENTITY_PATTERN_HE.search(text) or IDENTITY_PATTERN_EN.search(text))


def identity_answer(text: str) -> str:
    return IDENTITY_ANSWER_HE if has_hebrew(text) else IDENTITY_ANSWER_EN


# ════════════════════════════════════════════════════════════════
#  ACTION / IMPERATIVE REQUESTS — "תעשה לי X" / "do X for me"
#  כשהמשתמש מבקש ביצוע ולא הסבר: אם יש כלי אמיתי (מחשבון/קוד) — הוא
#  יריץ אותו; אם אין — נגיד "אני לא יודע" במקום להסביר את משמעות המילים.
# ════════════════════════════════════════════════════════════════

ACTION_VERBS_HE = {
    "עשה", "עשי", "תעשה", "תעשי", "בצע", "בצעי", "תבצע", "תבצעי",
    "סדר", "סדרי", "תסדר", "תסדרי", "הכן", "הכיני", "תכין", "תכיני",
    "בנה", "בני", "תבנה", "תבני", "תקן", "תקני", "תתקן", "תתקני",
    "פתח", "פתחי", "תפתח", "תפתחי", "סגור", "סגרי", "תסגור", "תסגרי",
    "שלח", "שלחי", "תשלח", "תשלחי", "מחק", "מחקי", "תמחק", "תמחקי",
    "הוסף", "הוסיפי", "תוסיף", "תוסיפי", "שנה", "שני", "תשנה", "תשני",
    "עדכן", "עדכני", "תעדכן", "תעדכני", "הורד", "הורידי", "תוריד", "תורידי",
    "שמור", "שמרי", "תשמור", "תשמרי",
}
ACTION_VERBS_EN = {
    "do", "make", "create", "perform", "run", "open", "close", "send",
    "delete", "remove", "add", "set", "build", "fix", "update", "save",
    "download", "install", "generate", "produce", "prepare", "organize",
}


def is_action_request(text: str) -> bool:
    """True אם המשתמש מנסח בקשה כפקודה (פועל ציווי) ולא כשאלת ידע."""
    norm = normalize_text(text)
    words = norm.split()
    if not words:
        return False
    first = words[0]
    return first in ACTION_VERBS_HE or first in ACTION_VERBS_EN


# ════════════════════════════════════════════════════════════════
#  CALCULATOR TOOL — "מודל מבין → כלי מחשב"
#  המודל לא מחשב ולא משנן. הוא *מבין* שזו בקשת חישוב ופולט קריאה מובנית
#  "<calc> op"; הכלי כאן מחלץ את שני המספרים מהקלט (תמיד מדויק) ומחשב
#  דטרמיניסטית לכל גודל. הבינה = ההבנה+ההחלטה; הכלי = הביצוע המדויק.
# ════════════════════════════════════════════════════════════════

CALC_MARKER = "<calc>"
_CALC_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def space_math_operators(text: str) -> str:
    """מוסיף רווחים סביב סמלי אופרטור כדי שהטוקנייזר (רמת-מילה) יראה '7 + 5'
    ולא טוקן יחיד '7+5'. נרמול קלט למודל בלבד — לא נוגע במשמעות."""
    return re.sub(r"([+\-*/])", r" \1 ", text)


def parse_calc_call(model_reply: str) -> str | None:
    """אם תשובת המודל היא קריאה לכלי ('<calc> *') — מחזיר את הפעולה שהמודל
    הבין (+,-,*,/). אחרת None. זו ההחלטה *הנלמדת* של המודל."""
    tokens = model_reply.strip().lower().split()
    if not tokens or tokens[0] != CALC_MARKER:
        return None
    for tok in tokens[1:]:
        if tok in _CALC_OPS:
            return tok
    return None


def execute_calc(user_text: str, op: str) -> str | None:
    """הכלי הדטרמיניסטי: מחלץ את שני המספרים הראשונים מהקלט המקורי (הם שם
    מילולית → מדויק תמיד) ומבצע את הפעולה שהמודל בחר. עובד לכל גודל מספר."""
    nums = re.findall(r"\d+", user_text)
    if len(nums) < 2 or op not in _CALC_OPS:
        return None
    a, b = int(nums[0]), int(nums[1])
    if op == "/" and b == 0:
        return "אי אפשר לחלק באפס." if has_hebrew(user_text) else "Cannot divide by zero."
    result = _CALC_OPS[op](a, b)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    result_text = str(result) if isinstance(result, int) else f"{result:.6g}"
    return f"התוצאה היא {result_text}." if has_hebrew(user_text) else f"The answer is {result_text}."


def token_confidence(user_text: str, tok: Any) -> float:
    tokens = tok.clean(user_text)
    if not tokens:
        return 0.0
    unk_id = tok.word2idx.get("<UNK>")
    known = sum(1 for token in tokens if tok.word2idx.get(token, unk_id) != unk_id)
    return known / len(tokens)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\u0590-\u05FF]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def load_source_documents() -> list[SourceDocument]:
    global SOURCE_CACHE
    if SOURCE_CACHE is not None:
        return SOURCE_CACHE
    docs: list[SourceDocument] = []
    if os.path.isdir(SOURCES_DIR):
        for filename in sorted(os.listdir(SOURCES_DIR)):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(SOURCES_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except OSError:
                continue
            if text:
                docs.append(SourceDocument(path=path, name=filename[:-4], text=text))
    SOURCE_CACHE = docs
    return docs


def score_source_document(query: str, doc: SourceDocument) -> float:
    query_norm = normalize_text(query)
    doc_norm = normalize_text(doc.name + " " + doc.text)
    if not query_norm or not doc_norm:
        return 0.0
    query_tokens = set(query_norm.split())
    doc_tokens = set(doc_norm.split())
    overlap = len(query_tokens & doc_tokens)
    score = float(overlap)
    if query_norm in doc_norm:
        score += 4.0
    if doc.name and normalize_text(doc.name) in query_norm:
        score += 5.0
    for token in query_tokens:
        if len(token) >= 4 and token in doc_norm:
            score += 0.5
    return score


# ════════════════════════════════════════════════════════════════
#  FIRST-PERSON STRIPPING — מקורות מקומיים כתובים בגוף ראשון
#  (למשל "X היה סבא שלי") מתארים קשר אישי של כותב המקור, לא של ה-AI,
#  אז פשוט מוציאים את המשפט הזה מהציטוט במקום לנסח אותו מחדש.
# ════════════════════════════════════════════════════════════════

FIRST_PERSON_MARKERS_HE = {
    "סבי", "סבתי", "אבי", "אמי", "בני", "בתי", "אחי", "אחותי",
    "דודי", "דודתי", "שלי",
}
_FIRST_PERSON_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in FIRST_PERSON_MARKERS_HE) + r")\b")


def strip_first_person_sentences(text: str) -> str:
    """מסיר משפטים המתארים קשר אישי-משפחתי, כדי שה-AI לא יטען אותם כשלו."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not _FIRST_PERSON_RE.search(s)]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def lookup_local_source(text: str) -> str | None:
    docs = load_source_documents()
    if not docs:
        return None
    best_doc = None
    best_score = 0.0
    for doc in docs:
        score = score_source_document(text, doc)
        if score > best_score:
            best_score = score
            best_doc = doc
    if not best_doc or best_score < 2.0:
        return None

    lines = [line.strip() for line in best_doc.text.splitlines() if line.strip()]
    if not lines:
        return None

    important_lines = []
    query_norm = normalize_text(text)
    for line in lines:
        line_norm = normalize_text(line)
        if any(token in line_norm for token in query_norm.split() if len(token) >= 3):
            important_lines.append(line)

    excerpt = " ".join(important_lines[:3]) if important_lines else " ".join(lines[:3])
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    excerpt = strip_first_person_sentences(excerpt)
    if not excerpt:
        return None
    prefix = "לפי המקור המקומי:" if has_hebrew(text) else "According to the local source:"
    return f"{prefix} {excerpt}"


# ════════════════════════════════════════════════════════════════
#  CREATIVE NAME SUGGESTIONS — "תן שמות יצירתיים ל<X>" / "creative names for <X>"
#  זיהוי כללי של *סוג* הבקשה (לא של "חנות לחם" ספציפית) + חילוץ הנושא X
#  מהטקסט, כדי שהכלי יעבוד לכל עסק/דבר ולא רק לדוגמה אחת ששוננה.
# ════════════════════════════════════════════════════════════════

NAMING_PATTERNS_HE = [
    r"שמות\s+יצירתי(?:ים)?\s+ל(.+)",
    r"שם\s+יצירתי\s+ל(.+)",
    r"תני?\s+לי\s+שמות\s+ל(.+)",
    r"רעיונות\s+לשמות?\s+ל(.+)",
]
NAMING_PATTERNS_EN = [
    r"creative names? for (.+)",
    r"suggest(?:ions)?\s+(?:a\s+)?names?\s+for (.+)",
    r"name ideas for (.+)",
]

NAME_TEMPLATES_HE = [
    "{x} הזהב",
    "פינת ה{x}",
    "{x} של פעם",
    "בית {x}",
    "{x} ומעלה",
]
NAME_TEMPLATES_EN = [
    "The Golden {x}",
    "{x} Corner",
    "Artisan {x}",
    "{x} House",
    "The Cozy {x}",
]


def extract_naming_subject(text: str) -> str | None:
    patterns = NAMING_PATTERNS_HE if has_hebrew(text) else NAMING_PATTERNS_EN
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip(" ?!.,")
            subject = re.sub(r"^(a|an|the)\s+", "", subject, flags=re.IGNORECASE)
            return subject if subject else None
    return None


def generate_creative_names(text: str) -> str | None:
    """מייצר הצעות שם גנריות סביב הנושא שחולץ מהטקסט — לא תלוי בנושא ספציפי."""
    subject = extract_naming_subject(text)
    if not subject:
        return None
    templates = NAME_TEMPLATES_HE if has_hebrew(text) else NAME_TEMPLATES_EN
    names = [t.format(x=subject) for t in templates]
    if has_hebrew(text):
        return "הנה כמה רעיונות לשם:\n" + "\n".join(f"• {n}" for n in names)
    return "Here are some name ideas:\n" + "\n".join(f"• {n}" for n in names)


# ════════════════════════════════════════════════════════════════
#  CODE GENERATOR — מחולל קוד Python אמיתי
# ════════════════════════════════════════════════════════════════

# מילות מפתח שמזהות בקשת קוד
CODE_TRIGGER_HE = {
    "קוד", "פונקציה", "תכתוב", "כתוב", "צור", "תן", "דוגמה", "דוגמא",
    "פייתון", "python", "לולאה", "תנאי", "רשימה", "מחלקה", "סקריפט",
    "כתבי", "צרי", "תני", "תכתבי",
}
CODE_TRIGGER_EN = {
    "code", "function", "write", "create", "give", "example", "python",
    "loop", "list", "class", "script", "make", "show", "generate",
}

# תבניות קוד מוכנות — ממיר כוונה → קוד אמיתי
CODE_TEMPLATES = {
    # פונקציות בסיסיות
    r"(פונקצי[הת]|function).*(חיבור|הוספ|sum|add)": (
        "def add(a, b):\n    return a + b\n\n# Example:\nprint(add(3, 5))  # 8"
    ),
    r"(פונקצי[הת]|function).*(כפל|מכפל|multiply|times)": (
        "def multiply(x, y):\n    return x * y\n\n# Example:\nprint(multiply(4, 6))  # 24"
    ),
    r"(פונקצי[הת]|function).*(חיסור|minus|subtract)": (
        "def subtract(a, b):\n    return a - b\n\n# Example:\nprint(subtract(10, 3))  # 7"
    ),
    r"(פונקצי[הת]|function|code|check).*(זוגי|even)": (
        "def is_even(n):\n    return n % 2 == 0\n\n# Examples:\nprint(is_even(4))  # True\nprint(is_even(7))  # False"
    ),
    r"(פונקצי[הת]|function).*(ראשוני|prime)": (
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\n# Examples:\nprint(is_prime(7))   # True\nprint(is_prime(10))  # False"
    ),
    r"(פונקצי[הת]|function).*(היפוך|reverse)": (
        "def reverse_string(s):\n    return s[::-1]\n\n# Example:\nprint(reverse_string('hello'))  # 'olleh'"
    ),
    r"(פונקצי[הת]|function).*(שלום|greet|greeting)": (
        "def greet(name):\n    return f'Hello, {name}!'\n\n# Example:\nprint(greet('Alice'))  # Hello, Alice!"
    ),
    r"(פונקצי[הת]|function).*(מקסימום|maximum|max)": (
        "def find_max(numbers):\n    return max(numbers)\n\n# Example:\nprint(find_max([3, 1, 7, 2]))  # 7"
    ),
    r"(פונקצי[הת]|function).*(מינימום|minimum|min)": (
        "def find_min(numbers):\n    return min(numbers)\n\n# Example:\nprint(find_min([3, 1, 7, 2]))  # 1"
    ),
    r"(פונקצי[הת]|function).*(ממוצע|average|mean)": (
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n# Example:\nprint(average([1, 2, 3, 4, 5]))  # 3.0"
    ),
    r"(פונקצי[הת]|function).*(factorial|עצרת)": (
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\n# Example:\nprint(factorial(5))  # 120"
    ),
    # לולאות
    r"(לולאה|loop).*(for|range|מספרים|numbers)": (
        "for i in range(1, 11):\n    print(i)"
    ),
    r"(לולאה|loop).*(while)": (
        "count = 0\nwhile count < 5:\n    print(count)\n    count += 1"
    ),
    r"(לולאה|loop|iterate).*(list|רשימה)": (
        "fruits = ['apple', 'banana', 'orange']\nfor fruit in fruits:\n    print(fruit)"
    ),
    r"(לולאה|loop).*(זוגי|even)": (
        "# Print even numbers from 1 to 20\nfor i in range(1, 21):\n    if i % 2 == 0:\n        print(i)"
    ),
    # רשימות
    r"(רשימה|list).*(מספרים|numbers|integers)": (
        "numbers = [1, 2, 3, 4, 5]\nprint(numbers)\nprint(sum(numbers))   # 15\nprint(max(numbers))   # 5"
    ),
    r"(רשימה|list).*(מיון|sort)": (
        "my_list = [5, 2, 8, 1, 9, 3]\nmy_list.sort()\nprint(my_list)  # [1, 2, 3, 5, 8, 9]\n\n# Or without modifying original:\nsorted_list = sorted(my_list)\nprint(sorted_list)"
    ),
    r"(רשימה|list).*(סינון|filter)": (
        "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nevens = [n for n in numbers if n % 2 == 0]\nprint(evens)  # [2, 4, 6, 8, 10]"
    ),
    # מחלקות / OOP
    r"(מחלקה|class).*(כלב|dog|animal|חיה)": (
        "class Dog:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def bark(self):\n        return f'{self.name} says: Woof!'\n\n# Usage:\ndog = Dog('Rex', 3)\nprint(dog.bark())  # Rex says: Woof!"
    ),
    r"(מחלקה|class).*(אדם|person|student|סטודנט)": (
        "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def introduce(self):\n        return f'Hi, I am {self.name} and I am {self.age} years old.'\n\n# Usage:\np = Person('Alice', 25)\nprint(p.introduce())"
    ),
    r"(מחלקה|class).*(מחשבון|calculator)": (
        "class Calculator:\n    def add(self, a, b): return a + b\n    def subtract(self, a, b): return a - b\n    def multiply(self, a, b): return a * b\n    def divide(self, a, b):\n        if b == 0:\n            return 'Error: division by zero'\n        return a / b\n\ncalc = Calculator()\nprint(calc.add(10, 5))       # 15\nprint(calc.multiply(4, 7))   # 28"
    ),
    # קבצים
    r"(קובץ|file).*(כתיב|כתוב|write)": (
        "# Writing to a file:\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello, World!\\n')\n    f.write('Second line\\n')\n\nprint('File written successfully.')"
    ),
    r"(קובץ|file).*(קרא|read)": (
        "# Reading from a file:\nwith open('output.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n\nprint(content)"
    ),
    # מילונים
    r"(מילון|dict|dictionary)": (
        "student = {\n    'name': 'Alice',\n    'age': 20,\n    'grade': 'A'\n}\n\nprint(student['name'])   # Alice\nprint(student.get('age'))  # 20\n\n# Iterating:\nfor key, value in student.items():\n    print(f'{key}: {value}')"
    ),
    # Fibonacci
    r"(fibonacci|פיבונאצ'י|fib)": (
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=' ')\n        a, b = b, a + b\n\nfibonacci(10)  # 0 1 1 2 3 5 8 13 21 34"
    ),
    # Hello World / שלום
    r"(שלום|hello).*(עולם|world)": (
        "print('Hello, World!')\nprint('שלום עולם!')"
    ),
    # Calculator / Fibonacci generic
    r"(מחשבון|calculator).*(פשוט|simple|basic)": (
        "def calculator(a, op, b):\n    if op == '+':\n        return a + b\n    elif op == '-':\n        return a - b\n    elif op == '*':\n        return a * b\n    elif op == '/':\n        return a / b if b != 0 else 'Error'\n    return 'Unknown operator'\n\nprint(calculator(10, '+', 5))   # 15\nprint(calculator(10, '*', 3))   # 30"
    ),
}

# תבניות ברירת מחדל לפי כוונה כללית
DEFAULT_CODE_BY_TOPIC = {
    "function": "def my_function(x):\n    result = x * 2\n    return result\n\nprint(my_function(5))  # 10",
    "loop":     "for i in range(5):\n    print(f'Step {i}')",
    "list":     "my_list = [10, 20, 30, 40, 50]\nfor item in my_list:\n    print(item)",
    "class":    "class MyClass:\n    def __init__(self, value):\n        self.value = value\n\n    def show(self):\n        print(f'Value: {self.value}')\n\nobj = MyClass(42)\nobj.show()  # Value: 42",
    "default":  "# Python example:\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('World'))  # Hello, World!",
}


def is_code_request(text: str) -> bool:
    """בודק אם המשתמש מבקש קוד."""
    lower = text.lower()
    words = set(re.findall(r"[\w\u0590-\u05FF]+", lower))

    has_he_trigger = bool(words & CODE_TRIGGER_HE)
    has_en_trigger = bool(words & CODE_TRIGGER_EN)

    # חייב להיות לפחות אחד
    if not (has_he_trigger or has_en_trigger):
        return False

    # מילות חרגה — שאלות ידע ולא בקשות קוד
    knowledge_words = {
        "מה", "מהי", "מהו", "what", "who", "מי", "explain", "הסבר",
        "הסברי", "תסביר", "why", "למה", "when", "מתי", "how", "איך",
    }
    # אם יש מילת ידע בלי מילת קוד חזקה — לא קוד
    if words & knowledge_words and not (words & {"כתוב", "write", "create", "צור", "תכתוב"}):
        return False

    return True


def generate_code(text: str) -> str | None:
    """מייצר קוד Python לפי הבקשה."""
    # אל תתעלם מבקשות יצירתי שמות — תן לעבור לדגם
    is_naming = any(re.search(p, text, re.IGNORECASE) for p in NAMING_PATTERNS_HE + NAMING_PATTERNS_EN)
    if is_naming:
        return None

    if not is_code_request(text):
        return None

    lower = text.lower()

    # חיפוש לפי תבניות ספציפיות
    for pattern, code in CODE_TEMPLATES.items():
        if re.search(pattern, lower):
            return f"```python\n{code}\n```"

    # ברירת מחדל לפי נושא כללי
    words = set(re.findall(r"[\w\u0590-\u05FF]+", lower))
    if words & {"פונקציה", "function", "def"}:
        return f"```python\n{DEFAULT_CODE_BY_TOPIC['function']}\n```"
    if words & {"לולאה", "loop", "for", "while"}:
        return f"```python\n{DEFAULT_CODE_BY_TOPIC['loop']}\n```"
    if words & {"רשימה", "list", "array"}:
        return f"```python\n{DEFAULT_CODE_BY_TOPIC['list']}\n```"
    if words & {"מחלקה", "class", "object", "אובייקט"}:
        return f"```python\n{DEFAULT_CODE_BY_TOPIC['class']}\n```"

    return f"```python\n{DEFAULT_CODE_BY_TOPIC['default']}\n```"


# ════════════════════════════════════════════════════════════════
#  WIKIPEDIA SEARCH
# ════════════════════════════════════════════════════════════════

QUESTION_PATTERNS = [
    r"^\s*מה זה\s+(.+?)\s*\??$",
    r"^\s*מי זה\s+(.+?)\s*\??$",
    r"^\s*מהי\s+(.+?)\s*\??$",
    r"^\s*what is\s+(.+?)\s*\??$",
    r"^\s*who is\s+(.+?)\s*\??$",
]


def extract_topic(text: str) -> str | None:
    low = text.strip().lower()
    for pattern in QUESTION_PATTERNS:
        match = re.match(pattern, low, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" ?!.")
            return topic if topic else None
    if token_confidence_like_topic(low):
        return low.strip(" ?!.")
    return None


def token_confidence_like_topic(text: str) -> bool:
    return len(text.split()) <= 4 and not re.search(r"\d|[\+\-\*/=]", text)


def request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("request failed")


def wikipedia_search_title(topic: str, lang: str) -> str | None:
    params = urllib.parse.urlencode({
        "action": "query", "format": "json",
        "list": "search", "srsearch": topic, "srlimit": "1",
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    data = request_json(url)
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    return results[0].get("title")


def wikipedia_summary(title: str, lang: str) -> str | None:
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    data = request_json(url)
    if data.get("type") == "disambiguation":
        return None
    extract = data.get("extract") or ""
    extract = re.sub(r"\s+", " ", extract).strip()
    if len(extract.split()) < 4:
        return None
    first_sentence = re.split(r"(?<=[.!?])\s+", extract)[0]
    return first_sentence[:500].strip()


def lookup_wikipedia(text: str) -> str | None:
    topic = extract_topic(text)
    if not topic:
        return None
    langs = ["he", "en"] if has_hebrew(text) else ["en", "he"]
    for lang in langs:
        try:
            title = wikipedia_search_title(topic, lang)
            if not title:
                continue
            summary = wikipedia_summary(title, lang)
        except Exception:
            continue
        if summary:
            return summary
    return None


def should_use_external_lookup(user_text: str, tok: Any) -> bool:
    confidence = token_confidence(user_text, tok)
    if confidence < 0.7:
        return True
    return extract_topic(user_text) is not None


# ════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINTS
#
#  הזרימה ב-chat.py:
#     quick_tool_reply → generate(model) → parse_calc_call/execute_calc
#                     → knowledge_reply
#  כלומר: החלטת-החישוב של *המודל* קודמת לחיפוש-ידע (מקור מקומי/ויקיפדיה),
#  כדי ששאלת חשבון לא תיחטף. אין יותר גארד-regex שמחליט "זו מתמטיקה" —
#  המודל הוא שמחליט (פולט <calc>), וזה בדיוק החלק הנלמד.
# ════════════════════════════════════════════════════════════════

def quick_tool_reply(user_text: str) -> str | None:
    """מענים ודאיים שרצים *לפני* המודל: שיחת חולין, זהות, קוד, בקשת-פעולה."""
    if is_greeting(user_text):
        return greeting_answer(user_text)

    if is_identity_question(user_text):
        return identity_answer(user_text)

    code = generate_code(user_text)
    if code:
        return code

    # בקשת פעולה (ציווי) שאין לה כלי בפועל — "לא יודע", לא הסבר מילים
    # אבל לא לבקשות יצירתיות כמו "תן שמות יצירתיים ל..." — תן הן תן לעבור לדגם
    is_naming = any(re.search(p, user_text, re.IGNORECASE) for p in NAMING_PATTERNS_HE + NAMING_PATTERNS_EN)
    if is_action_request(user_text) and not is_naming:
        return unknown_answer(user_text)

    return None


def knowledge_reply(user_text: str, tok: Any) -> str | None:
    """חיפוש-ידע שרץ *אחרי* שהמודל לא ביקש חישוב: מקור מקומי, ויקיפדיה,
    ולבסוף 'אני לא יודע' כשאין ביטחון."""
    local_source = lookup_local_source(user_text)
    if local_source:
        return local_source

    topic = extract_topic(user_text)
    if should_use_external_lookup(user_text, tok):
        wiki = lookup_wikipedia(user_text)
        if wiki:
            return wiki
        if topic is not None or token_confidence(user_text, tok) < 0.7:
            return unknown_answer(user_text)

    return None


def answer_with_tools(user_text: str, tok: Any) -> str | None:
    """נשמר לתאימות/שימוש עצמאי: quick ואז knowledge (בלי מסלול המודל)."""
    quick = quick_tool_reply(user_text)
    if quick is not None:
        return quick
    return knowledge_reply(user_text, tok)
