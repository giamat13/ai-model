"""
chat.py — ממשק גרפי לשיחה עם המודל המאומן
"""

import os
import re
import sys
import tkinter as tk
from tkinter import scrolledtext
import numpy as np

from tokenizer import Tokenizer
from model     import MiniLM
from assistant_tools import answer_with_tools, unknown_answer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── RTL fix ────────────────────────────────────────────────────────────────
def has_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)

def fix_rtl(text: str) -> str:
    lines = text.split("\n")
    fixed = []
    for line in lines:
        words = line.split()
        fixed.append(" ".join(reversed(words)) if words and has_hebrew(line) else line)
    return "\n".join(fixed)


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
    data   = np.load(model_path, allow_pickle=True)
    config = data["config"]
    vocab_size, embed_dim, hidden_dim, context_len = (
        int(config[0]), int(config[1]), int(config[2]), int(config[3])
    )
    model = MiniLM(vocab_size, embed_dim, hidden_dim, context_len)
    model.E  = data["E"]
    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]
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

    generated = list(base_ids)
    for _ in range(n_words):
        probs   = model.forward(np.array(ctx))
        logits  = np.log(probs + 1e-9) / max(temperature, 0.01)
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ctx = ctx[1:] + [next_id]

    full = tok.decode(generated, skip_special=True)
    marker = "model :"
    idx = full.find(marker)
    if idx != -1:
        reply = full[idx + len(marker):].strip()
        if "user :" in reply:
            reply = reply[:reply.index("user :")].strip()
        return reply
    return full


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
                log_write("ai_text", fix_rtl(content.strip()) + "\n")
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
        log_write("you_text", fix_rtl(user_text) + "\n")

        tool_reply = answer_with_tools(user_text, tok)
        if tool_reply:
            log_ai_reply(tool_reply)
            conversation_history.append(("user",  user_text))
            conversation_history.append(("model", tool_reply))
            return

        conversation_history.append(("user", user_text))
        prompt_parts = []
        for role, msg in conversation_history:
            prompt_parts.append(f"{role} : {msg}")
        prompt_parts.append("model :")
        full_prompt = " ".join(prompt_parts)

        tokens = tok.clean(user_text)
        unk_token = tok.word2idx["<UNK>"]
        unknown_count = sum(tok.word2idx.get(w, unk_token) == unk_token for w in tokens)
        if tokens and unknown_count / len(tokens) > 0.35:
            reply = unknown_answer(user_text)
        else:
            reply = generate(model, tok, full_prompt,
                             n_words=words_var.get(),
                             temperature=temp_var.get())

        if not reply:
            reply = "..."

        conversation_history.append(("model", reply))
        log_ai_reply(reply)

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
