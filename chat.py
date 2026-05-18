"""
chat.py — ממשק גרפי לשיחה עם המודל המאומן
"""

import os
import tkinter as tk
from tkinter import scrolledtext
import numpy as np

from tokenizer import Tokenizer
from model     import MiniLM

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── תיקון RTL: הופך סדר מילים בעברית ──────────────────────────────────────
def fix_rtl(text: str) -> str:
    """
    tkinter ב-Windows לא תומך ב-RTL.
    הפתרון: להפוך את סדר המילים בכל שורה.
    """
    lines = text.split("\n")
    fixed = []
    for line in lines:
        words = line.split()
        fixed.append(" ".join(reversed(words)) if words else "")
    return "\n".join(fixed)


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
    root.title("Mini Hebrew AI - Chat")
    root.geometry("680x540")
    root.configure(bg="#1e1e2e")
    root.resizable(True, True)

    # ── chat log ──
    log = scrolledtext.ScrolledText(
        root, state=tk.DISABLED,
        bg="#11111b", fg="#cdd6f4",
        font=("Consolas", 12),
        relief=tk.FLAT, bd=0,
        wrap=tk.WORD, padx=10, pady=10,
    )
    log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
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

    log_write("hint",    "Mini Hebrew AI - powered by NumPy only\n")
    log_write("hint",    "Type anything in Hebrew and press Enter\n")
    log_write("divider", "─" * 60 + "\n\n")

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

        # הצג שאלת משתמש — הפוך RTL
        log_write("you_lbl",  "You:   ")
        log_write("you_text", fix_rtl(user_text) + "\n")

        # בנה פרומפט בפורמט שהמודל מכיר
        prompt = f"user : {user_text} model :"
        known  = [w for w in tok.clean(prompt) if w in tok.word2idx]
        if len(known) < 2:
            log_write("hint",
                "AI:    " + fix_rtl("מילה לא מוכרת — נסה: שלום / מה זה / מי אתה") + "\n\n")
            return

        tokens = tok.clean(user_text)
        unk_token = tok.word2idx["<UNK>"]
        unknown_count = sum(tok.word2idx.get(w, unk_token) == unk_token for w in tokens)
        if tokens and unknown_count == len(tokens):
            reply = "אני לא יודע אם צריך."
        else:
            reply = generate(model, tok, prompt,
                             n_words=words_var.get(),
                             temperature=temp_var.get())

        if not reply:
            reply = "..."

        # הצג תשובה — הפוך RTL
        log_write("ai_lbl",  "AI:    ")
        log_write("ai_text", fix_rtl(reply) + "\n\n")

    entry.bind("<Return>", send)

    tk.Button(
        input_frame, text="Send  Enter",
        bg="#89b4fa", fg="#1e1e2e",
        activebackground="#74c7ec",
        font=("Consolas", 12, "bold"),
        relief=tk.FLAT, bd=0, padx=16,
        command=send, cursor="hand2",
    ).pack(side=tk.RIGHT, padx=(4, 8), pady=4, ipady=6)

    root.mainloop()


if __name__ == "__main__":
    main()
