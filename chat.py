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


def generate(model, tok, seed_text, n_words, temperature):
    pad_id = tok.word2idx["<PAD>"]
    bos_id = tok.word2idx["<BOS>"]
    eos_id = tok.word2idx["<EOS>"]

    base_ids = tok.encode(seed_text, add_special=False)
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

    return tok.decode(generated, skip_special=True)


def main():
    print("Loading model...")
    model, tok = load_model()
    print(f"Model loaded — {model.num_params():,} params, vocab {tok.vocab_size}")

    root = tk.Tk()
    root.title("Mini Language Model Chat")
    root.geometry("640x520")
    root.configure(bg="#1e1e2e")

    # chat log
    log = scrolledtext.ScrolledText(
        root, state=tk.DISABLED,
        bg="#11111b", fg="#cdd6f4",
        font=("Consolas", 12),
        relief=tk.FLAT, bd=8,
        wrap=tk.WORD,
    )
    log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    log.tag_config("you",   foreground="#89b4fa", font=("Consolas", 12, "bold"))
    log.tag_config("model", foreground="#a6e3a1", font=("Consolas", 12, "bold"))
    log.tag_config("text",  foreground="#cdd6f4", font=("Consolas", 12))
    log.tag_config("hint",  foreground="#585b70", font=("Consolas", 11))

    def log_write(tag, text):
        log.config(state=tk.NORMAL)
        log.insert(tk.END, text, tag)
        log.config(state=tk.DISABLED)
        log.see(tk.END)

    log_write("hint", "Model ready. Type a Hebrew word and press Send.\n")
    log_write("hint", "Try: bina / reshet / limud / makhshev / neuron\n\n")

    # sliders
    sliders_frame = tk.Frame(root, bg="#1e1e2e")
    sliders_frame.pack(fill=tk.X, padx=10)

    tk.Label(sliders_frame, text="Words:", bg="#1e1e2e", fg="#585b70",
             font=("Consolas", 10)).pack(side=tk.LEFT)
    words_var = tk.IntVar(value=12)
    tk.Scale(sliders_frame, from_=4, to=40, variable=words_var,
             orient=tk.HORIZONTAL, length=100,
             bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
             highlightthickness=0, relief=tk.FLAT,
             font=("Consolas", 9)).pack(side=tk.LEFT, padx=(2, 16))

    tk.Label(sliders_frame, text="Creativity:", bg="#1e1e2e", fg="#585b70",
             font=("Consolas", 10)).pack(side=tk.LEFT)
    temp_var = tk.DoubleVar(value=0.8)
    tk.Scale(sliders_frame, from_=0.1, to=2.0, resolution=0.1,
             variable=temp_var, orient=tk.HORIZONTAL, length=120,
             bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
             highlightthickness=0, relief=tk.FLAT,
             font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)

    # input row
    input_frame = tk.Frame(root, bg="#45475a", pady=2)
    input_frame.pack(fill=tk.X, padx=10, pady=10)

    entry = tk.Entry(
        input_frame,
        bg="#1e1e2e", fg="#ffffff",
        insertbackground="#ffffff",
        font=("Consolas", 14),
        relief=tk.SUNKEN, bd=3,
    )
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7, padx=4, pady=4)
    entry.focus_force()

    def send(event=None):
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, tk.END)

        log_write("you",  "You:   ")
        log_write("text", text + "\n")

        known = [w for w in tok.clean(text) if w in tok.word2idx]
        if not known:
            log_write("hint", "       (word not in vocabulary)\n\n")
            return

        result = generate(model, tok, text,
                          n_words=words_var.get(),
                          temperature=temp_var.get())

        log_write("model", "Model: ")
        log_write("text",  result + "\n\n")

    entry.bind("<Return>", send)

    tk.Button(
        input_frame, text="Send",
        bg="#89b4fa", fg="#1e1e2e",
        activebackground="#74c7ec",
        font=("Consolas", 12, "bold"),
        relief=tk.FLAT, bd=0, padx=14,
        command=send, cursor="hand2",
    ).pack(side=tk.RIGHT, padx=(6, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
