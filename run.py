"""
run.py — מאמן אם צריך, ואז פותח את הצ'אט.

הרצה:
    python run.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.npz")
TOK_PATH   = os.path.join(_HERE, "tokenizer.json")


def need_training() -> bool:
    return not os.path.exists(MODEL_PATH) or not os.path.exists(TOK_PATH)


def train():
    import json, time
    import numpy as np
    from tokenizer import Tokenizer
    from model     import MiniLM

    DATA_PATH   = os.path.join(_HERE, "training_data.json")
    CONTEXT_LEN = 5
    EMBED_DIM   = 64
    HIDDEN_DIM  = 128
    EPOCHS      = 150
    LR          = 0.08

    print("=" * 56)
    print("  אימון מודל — NumPy בלבד")
    print("=" * 56)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [a["text"] for a in data["articles"]]
    print(f"[Data] {len(texts)} מאמרים ודיאלוגים נטענו")

    tok = Tokenizer()
    tok.build_vocab(texts, min_freq=1)
    tok.save(TOK_PATH)

    # הכנת דוגמאות
    pad_id = tok.word2idx["<PAD>"]
    samples = []
    for text in texts:
        ids = tok.encode(text, add_special=True)
        if len(ids) <= CONTEXT_LEN:
            continue
        for i in range(len(ids) - CONTEXT_LEN):
            ctx    = ids[i : i + CONTEXT_LEN]
            target = ids[i + CONTEXT_LEN]
            if target != pad_id:
                samples.append((np.array(ctx), target))
    print(f"[Data] {len(samples):,} דוגמאות אימון")

    np.random.seed(42)
    model = MiniLM(tok.vocab_size, EMBED_DIM, HIDDEN_DIM, CONTEXT_LEN)
    print(f"[Model] {model.num_params():,} פרמטרים\n")

    print(f"{'Epoch':>6}  {'Loss':>8}  {'זמן':>5}")
    print("-" * 26)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        np.random.shuffle(samples)
        total = 0.0
        for ctx, target in samples:
            probs  = model.forward(ctx)
            loss   = model.cross_entropy_loss(probs, target)
            grads  = model.backward(target)
            model.update(grads, lr=LR)
            total += loss
        avg = total / len(samples)
        if epoch % 30 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg:>8.4f}  {time.time()-t0:>4.1f}s")

    np.savez(MODEL_PATH,
        E=model.E, W1=model.W1, b1=model.b1,
        W2=model.W2, b2=model.b2,
        config=np.array([tok.vocab_size, EMBED_DIM, HIDDEN_DIM, CONTEXT_LEN])
    )
    print(f"\nמודל נשמר → {MODEL_PATH}\n")


def chat():
    import tkinter as tk
    from tkinter import scrolledtext
    import numpy as np
    from tokenizer import Tokenizer
    from model     import MiniLM

    # טעינה
    tok = Tokenizer()
    tok.load(TOK_PATH)
    data   = np.load(MODEL_PATH, allow_pickle=True)
    config = data["config"]
    vocab_size, embed_dim, hidden_dim, context_len = (
        int(config[0]), int(config[1]), int(config[2]), int(config[3])
    )
    model = MiniLM(vocab_size, embed_dim, hidden_dim, context_len)
    model.E  = data["E"];  model.W1 = data["W1"];  model.b1 = data["b1"]
    model.W2 = data["W2"]; model.b2 = data["b2"]
    print(f"מודל נטען — {model.num_params():,} פרמטרים, vocab {tok.vocab_size}")

    # RTL fix
    def fix_rtl(text):
        return "\n".join(
            " ".join(reversed(line.split())) if line.strip() else ""
            for line in text.split("\n")
        )

    # יצירת טקסט
    def generate(user_text, n_words, temperature):
        pad_id = tok.word2idx["<PAD>"]
        bos_id = tok.word2idx["<BOS>"]
        eos_id = tok.word2idx["<EOS>"]
        prompt   = f"user : {user_text} model :"
        base_ids = tok.encode(prompt, add_special=False) or [bos_id]
        ctx      = base_ids[-context_len:]
        while len(ctx) < context_len:
            ctx = [pad_id] + ctx
        generated = list(base_ids)
        for _ in range(n_words):
            probs   = model.forward(np.array(ctx))
            logits  = np.log(probs + 1e-9) / max(temperature, 0.01)
            logits -= logits.max()
            probs   = np.exp(logits); probs /= probs.sum()
            nxt = np.random.choice(len(probs), p=probs)
            if nxt == eos_id:
                break
            generated.append(nxt); ctx = ctx[1:] + [nxt]
        full   = tok.decode(generated, skip_special=True)
        marker = "model :"
        idx    = full.find(marker)
        if idx != -1:
            reply = full[idx + len(marker):].strip()
            if "user :" in reply:
                reply = reply[:reply.index("user :")].strip()
            return reply
        return full

    # GUI
    root = tk.Tk()
    root.title("Mini Hebrew AI")
    root.geometry("680x540")
    root.configure(bg="#1e1e2e")
    root.resizable(True, True)

    log = scrolledtext.ScrolledText(
        root, state=tk.DISABLED,
        bg="#11111b", fg="#cdd6f4",
        font=("Consolas", 12), relief=tk.FLAT, bd=0,
        wrap=tk.WORD, padx=10, pady=10,
    )
    log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
    log.tag_config("you_lbl",  foreground="#89b4fa", font=("Consolas", 11, "bold"))
    log.tag_config("you_text", foreground="#cdd6f4", font=("Consolas", 12))
    log.tag_config("ai_lbl",   foreground="#a6e3a1", font=("Consolas", 11, "bold"))
    log.tag_config("ai_text",  foreground="#ffffff",  font=("Consolas", 12))
    log.tag_config("hint",     foreground="#585b70", font=("Consolas", 10))

    def log_write(tag, text):
        log.config(state=tk.NORMAL)
        log.insert(tk.END, text, tag)
        log.config(state=tk.DISABLED)
        log.see(tk.END)

    log_write("hint", "Mini Hebrew AI — NumPy only\n")
    log_write("hint", "─" * 55 + "\n\n")

    # sliders
    ctrl = tk.Frame(root, bg="#181825", pady=4)
    ctrl.pack(fill=tk.X, padx=10, pady=(4, 0))
    tk.Label(ctrl, text="Words:", bg="#181825", fg="#6c7086",
             font=("Consolas", 10)).pack(side=tk.LEFT, padx=(6, 2))
    words_var = tk.IntVar(value=14)
    tk.Scale(ctrl, from_=4, to=40, variable=words_var, orient=tk.HORIZONTAL,
             length=90, bg="#181825", fg="#cdd6f4", troughcolor="#313244",
             highlightthickness=0, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 12))
    tk.Label(ctrl, text="Creativity:", bg="#181825", fg="#6c7086",
             font=("Consolas", 10)).pack(side=tk.LEFT, padx=(0, 2))
    temp_var = tk.DoubleVar(value=0.6)
    tk.Scale(ctrl, from_=0.1, to=2.0, resolution=0.1, variable=temp_var,
             orient=tk.HORIZONTAL, length=110, bg="#181825", fg="#cdd6f4",
             troughcolor="#313244", highlightthickness=0, relief=tk.FLAT).pack(side=tk.LEFT)

    # input
    input_frame = tk.Frame(root, bg="#313244")
    input_frame.pack(fill=tk.X, padx=10, pady=8)
    entry = tk.Entry(input_frame, bg="#1e1e2e", fg="#ffffff",
                     insertbackground="#89b4fa", font=("Consolas", 14),
                     relief=tk.FLAT, bd=0)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=9, padx=(10, 4), pady=4)
    entry.focus_force()

    def send(event=None):
        user_text = entry.get().strip()
        if not user_text:
            return
        entry.delete(0, tk.END)
        log_write("you_lbl",  "You:   ")
        log_write("you_text", fix_rtl(user_text) + "\n")
        known = [w for w in tok.clean(f"user : {user_text} model :") if w in tok.word2idx]
        if len(known) < 2:
            log_write("hint", "AI:    " + fix_rtl("מילה לא מוכרת — נסה: שלום / מה זה / מי אתה") + "\n\n")
            return
        tokens = tok.clean(user_text)
        unk_token = tok.word2idx["<UNK>"]
        unknown_count = sum(tok.word2idx.get(w, unk_token) == unk_token for w in tokens)
        if tokens and unknown_count == len(tokens):
            reply = "אני לא יודע אם צריך."
        else:
            reply = generate(user_text, words_var.get(), temp_var.get()) or "..."
        log_write("ai_lbl",  "AI:    ")
        log_write("ai_text", fix_rtl(reply) + "\n\n")

    entry.bind("<Return>", send)
    tk.Button(input_frame, text="Send  Enter",
              bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
              font=("Consolas", 12, "bold"), relief=tk.FLAT, bd=0, padx=16,
              command=send, cursor="hand2").pack(side=tk.RIGHT, padx=(4, 8), pady=4, ipady=6)

    root.mainloop()


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if need_training():
        print("לא נמצא מודל — מתחיל אימון...\n")
        train()
    else:
        print("מודל קיים — מדלג על אימון.")
        print("(למחוק model.npz ו-tokenizer.json כדי לאמן מחדש)\n")

    print("פותח צ'אט...")
    chat()
