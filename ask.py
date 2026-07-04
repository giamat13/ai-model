"""
ask.py — בדיקת-שפיות ל-CLI: מריץ את *אותה* צנרת בדיוק כמו chat.py, בלי GUI.

שימוש:
  python ask.py "כמה זה 1+1" "כמה זה 3+6" "9 כפול 9"
  python ask.py            # מצב אינטראקטיבי (שורה אחר שורה)

נועד לוודא שהקוד+המודל שעל הדיסק עובדים, כשה-GUI מריץ אולי תהליך ישן.
"""

import sys

import chat as C
from assistant_tools import (
    quick_tool_reply, knowledge_reply,
    parse_calc_call, execute_calc, space_math_operators,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

model, tok = C.load_model()
history: list[tuple[str, str]] = []


def answer(user_text: str, n_words: int = 14, temperature: float = 0.6) -> str:
    """העתק מדויק של הזרימה ב-chat.py send()."""
    quick = quick_tool_reply(user_text)
    if quick:
        history.append(("user", user_text))
        history.append(("model", quick))
        return quick

    history.append(("user", user_text))
    prompt_parts = [f"{role} : {msg}" for role, msg in history] + ["model :"]
    full_prompt = space_math_operators(" ".join(prompt_parts))
    raw = C.generate(model, tok, full_prompt, n_words=n_words, temperature=temperature)

    op = parse_calc_call(raw)
    if op:
        reply = execute_calc(user_text, op) or knowledge_reply(user_text, tok) or raw
    else:
        reply = knowledge_reply(user_text, tok) or raw or "..."
    history.append(("model", reply))
    return reply


def main() -> None:
    queries = sys.argv[1:]
    if queries:
        for q in queries:
            print(f"You: {q}")
            print(f"AI:  {answer(q)}\n")
        return
    print("מצב אינטראקטיבי — הקלד שאלה (Ctrl+C ליציאה):")
    try:
        while True:
            q = input("You: ").strip()
            if q:
                print(f"AI:  {answer(q)}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
