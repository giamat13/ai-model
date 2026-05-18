"""
assistant_tools.py - כלים חיצוניים למודל הזעיר.

המודל עצמו קטן ולכן הוא לא אמור לנחש בכוח. הקובץ הזה מוסיף:
  * מחשבון בטוח לביטויים מתמטיים פשוטים.
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


CALC_WORDS = {
    "כמה",
    "חשב",
    "חשבי",
    "תחשב",
    "תחשבי",
    "חישוב",
    "מתמטיקה",
    "calculate",
    "calc",
    "compute",
    "solve",
}

MATH_SYMBOL_RE = re.compile(r"\d\s*[\+\-\*/\^%]\s*\d|[\(\)]")


def should_calculate(text: str) -> bool:
    cleaned = text.lower()
    words = set(re.findall(r"[\w\u0590-\u05FF]+", cleaned))
    return bool(MATH_SYMBOL_RE.search(cleaned) or (words & CALC_WORDS and re.search(r"\d", cleaned)))


def normalize_expression(text: str) -> str:
    expr = text.lower()
    replacements = {
        "כמה זה": "",
        "מה זה": "",
        "חשב": "",
        "חשבי": "",
        "תחשב": "",
        "תחשבי": "",
        "calculate": "",
        "compute": "",
        "solve": "",
        "כפול": "*",
        "לחלק": "/",
        "חלקי": "/",
        "ועוד": "+",
        "עוד": "+",
        "פלוס": "+",
        "מינוס": "-",
        "פחות": "-",
        "בחזקת": "**",
        "^": "**",
    }
    for src, dst in replacements.items():
        expr = expr.replace(src, dst)
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r"[^0-9\.\+\-\*/%\(\)\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
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
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": topic,
            "srlimit": "1",
        }
    )
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


def answer_with_tools(user_text: str, tok: Any) -> str | None:
    calc = calculate_answer(user_text)
    if calc:
        return calc

    topic = extract_topic(user_text)
    if should_use_external_lookup(user_text, tok):
        wiki = lookup_wikipedia(user_text)
        if wiki:
            return wiki
        if topic is not None or token_confidence(user_text, tok) < 0.7:
            return unknown_answer(user_text)

    return None
