"""
assistant_tools.py - כלים חיצוניים למודל הזעיר.

המודל עצמו קטן ולכן הוא לא אמור לנחש בכוח. הקובץ הזה מוסיף:
  * מחשבון בטוח לביטויים מתמטיים פשוטים.
  * מחולל קוד Python חכם לפקודות כתיבת קוד.
  * חיפוש קצר בויקיפדיה לשאלות ידע.
  * החלטת "אני לא יודע" כאשר אין מספיק מידע.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import json
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


UNKNOWN_HE = "אני לא יודע."
UNKNOWN_EN = "I don't know."
USER_AGENT = "mini-bilingual-ai-assistant/1.0"


def has_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def unknown_answer(text: str) -> str:
    return UNKNOWN_HE if has_hebrew(text) else UNKNOWN_EN


def token_confidence(user_text: str, tok: Any) -> float:
    tokens = tok.clean(user_text)
    if not tokens:
        return 0.0
    unk_id = tok.word2idx.get("<UNK>")
    known = sum(1 for token in tokens if tok.word2idx.get(token, unk_id) != unk_id)
    return known / len(tokens)


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
#  CALCULATOR
# ════════════════════════════════════════════════════════════════

CALC_WORDS = {
    "כמה", "חשב", "חשבי", "תחשב", "תחשבי", "חישוב", "מתמטיקה",
    "calculate", "calc", "compute", "solve",
}
MATH_SYMBOL_RE = re.compile(r"\d\s*[\+\-\*/\^%]\s*\d|[\(\)]")


def should_calculate(text: str) -> bool:
    cleaned = text.lower()
    words = set(re.findall(r"[\w\u0590-\u05FF]+", cleaned))
    return bool(MATH_SYMBOL_RE.search(cleaned) or (words & CALC_WORDS and re.search(r"\d", cleaned)))


def normalize_expression(text: str) -> str:
    expr = text.lower()
    replacements = {
        "כמה זה": "", "מה זה": "", "חשב": "", "חשבי": "",
        "תחשב": "", "תחשבי": "", "calculate": "", "compute": "", "solve": "",
        "כפול": "*", "לחלק": "/", "חלקי": "/", "ועוד": "+",
        "עוד": "+", "פלוס": "+", "מינוס": "-", "פחות": "-",
        "בחזקת": "**", "^": "**",
    }
    for src, dst in replacements.items():
        expr = expr.replace(src, dst)
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r"[^0-9\.\+\-\*/%\(\)\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent too large")
        return ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
        return ALLOWED_UNARY[type(node.op)](_eval_math(node.operand))
    raise ValueError("Unsupported expression")


def calculate_answer(text: str) -> str | None:
    if not should_calculate(text):
        return None
    expr = normalize_expression(text)
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_math(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return unknown_answer(text)

    if math.isfinite(result) and result.is_integer():
        result_text = str(int(result))
    else:
        result_text = f"{result:.10g}"
    return f"התוצאה היא {result_text}." if has_hebrew(text) else f"The answer is {result_text}."


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
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def answer_with_tools(user_text: str, tok: Any) -> str | None:
    # 1. מחשבון — ראשון
    calc = calculate_answer(user_text)
    if calc:
        return calc

    # 2. מחולל קוד — לפני ויקיפדיה!
    code = generate_code(user_text)
    if code:
        return code

    # 3. ויקיפדיה + אני לא יודע
    topic = extract_topic(user_text)
    if should_use_external_lookup(user_text, tok):
        wiki = lookup_wikipedia(user_text)
        if wiki:
            return wiki
        if topic is not None or token_confidence(user_text, tok) < 0.7:
            return unknown_answer(user_text)

    return None
