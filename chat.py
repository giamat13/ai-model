"""
chat.py — ממשק גרפי לשיחה עם המודל המאומן
"""

import os
import re
import sys
import textwrap
import tkinter as tk
import tkinter.font as tkfont
from tkinter import scrolledtext
import numpy as np

from tokenizer import Tokenizer
from model     import MiniLM
from assistant_tools import (
    quick_tool_reply, knowledge_reply,
    parse_calc_call, execute_calc, space_math_operators,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── RTL fix ────────────────────────────────────────────────────────────────
def has_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)

def fix_rtl(text: str, width: int = 70) -> str:
    """
    מתקן כיווניות RTL להצגה ב-Tk (שאינו תומך ב-bidi).
    חשוב: קודם עוטפים כל פסקה לשורות-תצוגה לפי `width`, ורק *אז* הופכים
    את סדר המילים בכל שורה-תצוגה בנפרד. אם היינו הופכים את כל הפסקה
    כמקשה אחת ומשאירים ל-Tk לעטוף אותה, סדר השורות (מלמעלה למטה) היה
    יוצא הפוך — השורה הראשונה שהוצגה הייתה מכילה את הסוף הלוגי של הטקסט.
    """
    out_lines = []
    for line in text.split("\n"):
        if not line.strip() or not has_hebrew(line):
            out_lines.append(line)
            continue
        for sub in (textwrap.wrap(line, width=width) or [line]):
            out_lines.append(" ".join(reversed(sub.split())))
    return "\n".join(out_lines)


# ── פיצול תשובה לחלקי טקסט וקוד ──────────────────────────────────────────
def split_reply(text: str) -> list[tuple[str, str]]:
    """
    מחזיר רשימה של (סוג, תוכן):
      ("text", "...")  — טקסט רגיל
      ("code", "...")  — בלוק קוד (ללא backticks)
    """
    parts = []
    pattern = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        before = text[last:m.start()]
        if before.strip():
            parts.append(("text", before))
        parts.append(("code", m.group(1).rstrip()))
        last = m.end()
    tail = text[last:]
    if tail.strip():
        parts.append(("text", tail))
    return parts if parts else [("text", text)]


# ══════════════════════════════════════════════════════
#  טעינת מודל
# ══════════════════════════════════════════════════════
def load_model():
    tok_path   = os.path.join(_HERE, "tokenizer.json")
    model_path = os.path.join(_HERE, "model.npz")
    tok = Tokenizer()
    tok.load(tok_path)
    # מקור-אמת יחיד לטעינה — אותו קוד ש-train.py משתמש בו (ראה MiniLM.load)
    model = MiniLM.load(model_path)
    if model is None:
        raise RuntimeError(
            "model.npz בפורמט ישן (ללא Attention). הרץ מחדש: python train.py"
        )
    return model, tok


# ══════════════════════════════════════════════════════
#  יצירת טקסט
# ══════════════════════════════════════════════════════
def generate(model, tok, prompt, n_words, temperature):
    pad_id = tok.word2idx["<PAD>"]
    bos_id = tok.word2idx["<BOS>"]
    eos_id = tok.word2idx["<EOS>"]

    base_ids = tok.encode(prompt, add_special=False)
    if not base_ids:
        base_ids = [bos_id]

    ctx = base_ids[-model.context_len:]
    while len(ctx) < model.context_len:
        ctx = [pad_id] + ctx

    new_ids = []
    for _ in range(n_words):
        probs   = model.forward(np.array(ctx))
        logits  = np.log(probs + 1e-9) / max(temperature, 0.01)
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        if next_id == eos_id:
            break
        new_ids.append(next_id)
        ctx = ctx[1:] + [next_id]

    # מפענחים רק את הטוקנים *החדשים* שהמודל ייצר — לא את כל הפרומפט. בשיחה
    # מרובת-תורים הפרומפט מכיל כמה "model :", ולכן חיפוש המרקר החזיר בעבר את
    # התשובה הישנה במקום את ההמשך החדש (וכך <calc> "נעלם" ושאלת חשבון נחטפה).
    reply = tok.decode(new_ids, skip_special=True).strip()
    for cut in ("user :", "model :"):
        if cut in reply:
            reply = reply[:reply.index(cut)].strip()
    return reply


# ══════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════
def main():
    print("Loading model...")
    model, tok = load_model()
    print(f"Ready — {model.num_params():,} params")

    root = tk.Tk()
    root.title("Mini Hebrew/English AI - Chat")
    root.geometry("700x580")
    root.configure(bg="#1e1e2e")
    root.resizable(True, True)

    # ── chat log ──
    log = scrolledtext.ScrolledText(
        root, state=tk.DISABLED,
        bg="#11111b", fg="#cdd6f4",
        font=("Consolas", 12),
        relief=tk.FLAT, bd=0,
        wrap=tk.WORD, padx=10, pady=10,
        spacing3=2,
    )
    log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

    # ── טאגים לטקסט רגיל ──
    log.tag_config("you_lbl",  foreground="#89b4fa", font=("Consolas", 11, "bold"))
    log.tag_config("you_text", foreground="#cdd6f4", font=("Consolas", 12))
    log.tag_config("ai_lbl",   foreground="#a6e3a1", font=("Consolas", 11, "bold"))
    log.tag_config("ai_text",  foreground="#ffffff", font=("Consolas", 12))
    log.tag_config("hint",     foreground="#585b70", font=("Consolas", 10))
    log.tag_config("divider",  foreground="#313244", font=("Consolas", 8))

    text_font = tkfont.Font(family="Consolas", size=12)

    def wrap_width_chars() -> int:
        """כמה תווים נכנסים כרגע בשורה אחת של הלוג, לפי הרוחב בפועל של החלון."""
        avail_px = log.winfo_width() - 40  # padx + שוליים
        char_px = text_font.measure("א") or 9
        if avail_px <= 0:
            return 70
        return max(20, avail_px // char_px)

    def log_write(tag, text):
        log.config(state=tk.NORMAL)
        log.insert(tk.END, text, tag)
        log.config(state=tk.DISABLED)
        log.see(tk.END)

    # ── הוספת בלוק קוד בתוך הלוג ──
    def log_code_block(code: str):
        """מוסיף קנבס קוד מעוצב inline בתוך ה-ScrolledText."""
        log.config(state=tk.NORMAL)

        lines      = code.split("\n")
        line_count = len(lines)
        max_chars  = max((len(l) for l in lines), default=40)

        # גובה ורוחב הקנבס
        ch   = ("Consolas", 11)
        char_w = 8          # pixels per char (קירוב ל-Consolas 11)
        char_h = 18         # pixels per line
        pad_x  = 12
        pad_y  = 8
        canvas_w = max(500, min(max_chars * char_w + pad_x * 2, 640))
        canvas_h = line_count * char_h + pad_y * 2 + 28  # +28 לשורת header

        canvas = tk.Canvas(
            log,
            width=canvas_w, height=canvas_h,
            bg="#1e1e2e", highlightthickness=1,
            highlightbackground="#45475a",
            relief=tk.FLAT, cursor="arrow",
        )

        # ── header bar ──
        canvas.create_rectangle(0, 0, canvas_w, 26, fill="#181825", outline="")
        canvas.create_oval(10, 8, 20, 18, fill="#f38ba8", outline="")
        canvas.create_oval(26, 8, 36, 18, fill="#f9e2af", outline="")
        canvas.create_oval(42, 8, 52, 18, fill="#a6e3a1", outline="")
        canvas.create_text(canvas_w // 2, 13,
                           text="Python", fill="#6c7086",
                           font=("Consolas", 9))

        # ── שורת הפרדה ──
        canvas.create_line(0, 26, canvas_w, 26, fill="#313244")

        # ── קוד עצמו — Syntax Highlight בסיסי ──
        KEYWORDS = {
            "def", "return", "if", "else", "elif", "for", "while",
            "in", "not", "and", "or", "import", "from", "class",
            "True", "False", "None", "print", "range", "self",
            "break", "continue", "pass", "lambda", "yield",
        }
        COLOR_KW      = "#cba6f7"   # סגול — keywords
        COLOR_STR     = "#a6e3a1"   # ירוק — strings
        COLOR_COMMENT = "#585b70"   # אפור — comments
        COLOR_NUM     = "#fab387"   # כתום — numbers
        COLOR_DEFAULT = "#cdd6f4"   # לבן-כחול — רגיל
        COLOR_FUNC    = "#89b4fa"   # כחול — function names after def/class
        COLOR_BUILTIN = "#89dceb"   # תכלת — builtins

        BUILTINS = {"print", "len", "range", "max", "min", "sum",
                    "int", "str", "float", "list", "dict", "set",
                    "sorted", "enumerate", "zip", "map", "filter"}

        def colorize_line(line: str) -> list[tuple[str, str]]:
            """מחזיר רשימה של (צבע, טקסט) לשורה אחת."""
            tokens = []
            # comment
            if "#" in line:
                pre, comment = line.split("#", 1)
                tokens.extend(colorize_line(pre))
                tokens.append((COLOR_COMMENT, "# " + comment))
                return tokens
            # strings (single & double)
            str_pat = re.compile(r"""('[^']*'|"[^"]*")""")
            parts = str_pat.split(line)
            for part in parts:
                if str_pat.match(part):
                    tokens.append((COLOR_STR, part))
                else:
                    # tokenize words
                    word_pat = re.compile(r"(\w+|[^\w]+)")
                    for seg in word_pat.findall(part):
                        if re.fullmatch(r"\d+(\.\d+)?", seg):
                            tokens.append((COLOR_NUM, seg))
                        elif seg in KEYWORDS:
                            tokens.append((COLOR_KW, seg))
                        elif seg in BUILTINS:
                            tokens.append((COLOR_BUILTIN, seg))
                        else:
                            tokens.append((COLOR_DEFAULT, seg))
            return tokens

        # ── ציור הקוד ──
        y = 26 + pad_y
        for line in lines:
            x = pad_x
            colored = colorize_line(line)
            for color, seg in colored:
                canvas.create_text(x, y, text=seg, anchor="nw",
                                   fill=color, font=("Consolas", 11))
                # הזזת x — נאמד לפי מספר תווים
                x += len(seg) * char_w
            y += char_h

        # ── כפתור Copy ──
        def copy_code(event=None):
            root.clipboard_clear()
            root.clipboard_append(code)
            btn_copy.config(text="✓ Copied!")
            root.after(1500, lambda: btn_copy.config(text="Copy"))

        btn_copy = tk.Button(
            canvas, text="Copy",
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a",
            font=("Consolas", 8), relief=tk.FLAT,
            bd=0, padx=6, pady=2,
            cursor="hand2", command=copy_code,
        )
        canvas.create_window(canvas_w - 6, 4, anchor="ne", window=btn_copy)

        # ── הכנסת הקנבס לתוך הטקסט ──
        log.window_create(tk.END, window=canvas, padx=6, pady=4)
        log.insert(tk.END, "\n\n")
        log.config(state=tk.DISABLED)
        log.see(tk.END)

    # ── פונקציה מאוחדת לכתיבת תשובת AI ──
    def log_ai_reply(reply: str):
        """מציגה תשובה — טקסט רגיל כטקסט, קוד כקנבס."""
        parts = split_reply(reply)
        log_write("ai_lbl", "AI:    ")
        for kind, content in parts:
            if kind == "code":
                log_write("ai_text", "\n")   # שורה לפני הקנבס
                log_code_block(content)
            else:
                log_write("ai_text", fix_rtl(content.strip(), wrap_width_chars()) + "\n")
        log_write("ai_text", "\n")

    log_write("hint",    "Mini Hebrew/English AI - powered by NumPy only\n")
    log_write("hint",    "Type in Hebrew or English and press Enter\n")
    log_write("divider", "─" * 60 + "\n\n")

    conversation_history = []

    # ── sliders ──
    ctrl = tk.Frame(root, bg="#181825", pady=4)
    ctrl.pack(fill=tk.X, padx=10, pady=(4, 0))

    tk.Label(ctrl, text="Words:", bg="#181825", fg="#6c7086",
             font=("Consolas", 10)).pack(side=tk.LEFT, padx=(6, 2))
    words_var = tk.IntVar(value=14)
    tk.Scale(ctrl, from_=4, to=40, variable=words_var,
             orient=tk.HORIZONTAL, length=90, showvalue=True,
             bg="#181825", fg="#cdd6f4", troughcolor="#313244",
             highlightthickness=0, relief=tk.FLAT,
             font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 12))

    tk.Label(ctrl, text="Creativity:", bg="#181825", fg="#6c7086",
             font=("Consolas", 10)).pack(side=tk.LEFT, padx=(0, 2))
    temp_var = tk.DoubleVar(value=0.6)
    tk.Scale(ctrl, from_=0.1, to=2.0, resolution=0.1,
             variable=temp_var, orient=tk.HORIZONTAL, length=110, showvalue=True,
             bg="#181825", fg="#cdd6f4", troughcolor="#313244",
             highlightthickness=0, relief=tk.FLAT,
             font=("Consolas", 9)).pack(side=tk.LEFT)

    # ── input row ──
    input_frame = tk.Frame(root, bg="#313244")
    input_frame.pack(fill=tk.X, padx=10, pady=8)

    entry = tk.Entry(
        input_frame,
        bg="#1e1e2e", fg="#ffffff",
        insertbackground="#89b4fa",
        font=("Consolas", 14),
        relief=tk.FLAT, bd=0,
    )
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
               ipady=9, padx=(10, 4), pady=4)
    entry.focus_force()

    def send(event=None):
        user_text = entry.get().strip()
        if not user_text:
            return
        entry.delete(0, tk.END)

        log_write("you_lbl",  "You:   ")
        log_write("you_text", fix_rtl(user_text, wrap_width_chars()) + "\n")
        print(f"You: {user_text}")

        # 1. מענים ודאיים לפני המודל (ברכה/זהות/קוד/בקשת-פעולה)
        quick = quick_tool_reply(user_text)
        if quick:
            log_ai_reply(quick)
            print(f"AI:  {quick}")
            conversation_history.append(("user",  user_text))
            conversation_history.append(("model", quick))
            return

        conversation_history.append(("user", user_text))

        # 2. המודל רץ ומחליט. הקלט מרווח סביב אופרטורים ("7+5"→"7 + 5") כדי
        #    שהטוקנייזר (רמת-מילה) יקרא נכון. אם המודל *הבין* שזו בקשת חישוב,
        #    הוא פולט קריאה מובנית "<calc> op".
        prompt_parts = [f"{role} : {msg}" for role, msg in conversation_history]
        prompt_parts.append("model :")
        full_prompt = space_math_operators(" ".join(prompt_parts))
        raw_reply = generate(model, tok, full_prompt,
                             n_words=words_var.get(),
                             temperature=temp_var.get())

        # 3. קריאה לכלי? → חישוב דטרמיניסטי מדויק (לכל גודל מספר)
        op = parse_calc_call(raw_reply)
        if op:
            result = execute_calc(user_text, op)
            reply = result or knowledge_reply(user_text, tok) or raw_reply
        else:
            # 4. לא בקשת חישוב → חיפוש-ידע (מקור/ויקיפדיה) עדיף על פטפוט; אחרת
            #    משאירים את תשובת המודל.
            reply = knowledge_reply(user_text, tok) or raw_reply or "..."

        conversation_history.append(("model", reply))
        log_ai_reply(reply)
        print(f"AI:  {reply}")

    def clear_chat():
        conversation_history.clear()
        log.config(state=tk.NORMAL)
        log.delete(1.0, tk.END)
        log.config(state=tk.DISABLED)
        log_write("hint",    "Mini Hebrew/English AI - powered by NumPy only\n")
        log_write("hint",    "Type in Hebrew or English and press Enter\n")
        log_write("divider", "─" * 60 + "\n\n")

    entry.bind("<Return>", send)

    tk.Button(
        input_frame, text="Send  Enter",
        bg="#89b4fa", fg="#1e1e2e",
        activebackground="#74c7ec",
        font=("Consolas", 12, "bold"),
        relief=tk.FLAT, bd=0, padx=16,
        command=send, cursor="hand2",
    ).pack(side=tk.RIGHT, padx=(4, 8), pady=4, ipady=6)

    tk.Button(
        input_frame, text="Clear",
        bg="#f38ba8", fg="#1e1e2e",
        activebackground="#eba0ac",
        font=("Consolas", 10, "bold"),
        relief=tk.FLAT, bd=0, padx=12,
        command=clear_chat, cursor="hand2",
    ).pack(side=tk.RIGHT, padx=(4, 4), pady=4, ipady=6)

    root.mainloop()


if __name__ == "__main__":
    main()
