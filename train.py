"""
train.py — לולאת אימון מלאה

מה קורה כאן:
  1. טוענים את הנתונים מה-JSON
  2. בונים Tokenizer
  3. מכינים דוגמאות: כל "חלון" של מילים + המילה הבאה שלו
  4. מריצים epochs — בכל epoch עוברים על כל הדוגמאות
  5. מציגים את ה-loss ודוגמת יצירה כל כמה צעדים
"""

import json
import os
import sys
import time
import numpy as np

from tokenizer import Tokenizer
from model     import MiniLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════
#  הגדרות אימון — שנה כאן כדי להתנסות
# ════════════════════════════════════════════════════════════════

import os
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(_HERE, "training_data.json")
API_DATA_PATH = os.path.join(_HERE, "api_training_data.json")
DATA_PATHS  = [DATA_PATH, API_DATA_PATH]
SAVE_DIR    = _HERE

CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "5"))       # כמה מילים אחורה המודל רואה
EMBED_DIM   = int(os.environ.get("EMBED_DIM", "64"))        # מימד ה-embedding לכל מילה
HIDDEN_DIM  = int(os.environ.get("HIDDEN_DIM", "128"))      # נוירונים בשכבה הנסתרת
EPOCHS      = int(os.environ.get("EPOCHS", "120"))          # כמה פעמים לעבור על כל הנתונים
LR          = float(os.environ.get("LR", "0.01"))           # learning rate
LOG_EVERY   = int(os.environ.get("LOG_EVERY", "10"))        # כל כמה epochs להדפיס דוגמה


# ════════════════════════════════════════════════════════════════
#  טעינת נתונים
# ════════════════════════════════════════════════════════════════

def load_texts(paths: list[str]) -> list[str]:
    texts = []
    loaded_files = []

    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        file_texts = [a["text"] for a in data.get("articles", []) if a.get("text")]
        texts.extend(file_texts)
        loaded_files.append((os.path.basename(path), len(file_texts)))

    if not texts:
        raise FileNotFoundError("לא נמצאו קבצי דאטה לאימון")

    summary = ", ".join(f"{name}: {count}" for name, count in loaded_files)
    print(f"[Data] נטענו {len(texts)} פריטים ({summary})")
    return texts


# ════════════════════════════════════════════════════════════════
#  הכנת דוגמאות אימון
# ════════════════════════════════════════════════════════════════

def make_samples(
    tok: Tokenizer, texts: list[str], context_len: int
) -> list[tuple[np.ndarray, int]]:
    """
    ממיר טקסטים לדוגמאות (context → next_word).
    
    לדוגמה עם context_len=3:
      טקסט: [2, 5, 8, 11, 4, 9]   (אחרי encode)
      דוגמאות:
        ([2, 5, 8], 11)
        ([5, 8, 11], 4)
        ([8, 11, 4], 9)
    """
    samples = []
    pad_id  = tok.word2idx["<PAD>"]

    for text in texts:
        ids = tok.encode(text, add_special=True)
        if len(ids) <= context_len:
            continue

        for i in range(len(ids) - context_len):
            ctx    = np.array(ids[i : i + context_len])
            target = ids[i + context_len]
            # דלג על דוגמאות שהמטרה שלהן היא PAD
            if target == pad_id:
                continue
            samples.append((ctx, target))

    print(f"[Data] דוגמאות אימון: {len(samples):,}")
    return samples


# ════════════════════════════════════════════════════════════════
#  יצירת טקסט — לבדיקה תוך-כדי אימון
# ════════════════════════════════════════════════════════════════

def generate(
    model:       MiniLM,
    tok:         Tokenizer,
    seed_text:   str,
    n_words:     int = 12,
    temperature: float = 0.8,
) -> str:
    """
    מקבל מילה/ביטוי ומייצר המשך.
    
    temperature:
      • גבוה (>1) → יצירתי יותר, פחות צפוי
      • נמוך (<1) → שמרני יותר, מילה נפוצה יותר
      • 1.0       → הסתברויות ישירות מהמודל
    """
    pad_id = tok.word2idx["<PAD>"]
    bos_id = tok.word2idx["<BOS>"]
    eos_id = tok.word2idx["<EOS>"]

    # קידוד הבסיס
    base_ids = tok.encode(seed_text, add_special=False)
    if not base_ids:
        base_ids = [bos_id]

    # ריפוד/חיתוך לאורך הנכון
    ctx = base_ids[-model.context_len:]
    while len(ctx) < model.context_len:
        ctx = [pad_id] + ctx

    generated = list(base_ids)

    for _ in range(n_words):
        ctx_arr = np.array(ctx)
        probs   = model.forward(ctx_arr)

        # החלת temperature
        logits  = np.log(probs + 1e-9) / temperature
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        # דגימה מהתפלגות (לא רק argmax — מגוון יותר)
        next_id = np.random.choice(len(probs), p=probs)

        if next_id == eos_id:
            break

        generated.append(next_id)
        ctx = ctx[1:] + [next_id]

    return tok.decode(generated, skip_special=True)


# ════════════════════════════════════════════════════════════════
#  לולאת אימון ראשית
# ════════════════════════════════════════════════════════════════

def train():
    print("=" * 56)
    print("  🧠  אימון מודל שפה זעיר — NumPy בלבד")
    print("=" * 56)

    # ── 1. נתונים ───────────────────────────────────────────────
    texts = load_texts(DATA_PATHS)

    # ── 2. Tokenizer ────────────────────────────────────────────
    tok = Tokenizer()
    tok.build_vocab(texts, min_freq=1)
    tok.save(f"{SAVE_DIR}/tokenizer.json")

    # ── 3. דוגמאות ──────────────────────────────────────────────
    samples = make_samples(tok, texts, CONTEXT_LEN)
    if not samples:
        print("❌ אין דוגמאות! בדוק את הנתונים.")
        return

    # ── 4. מודל ─────────────────────────────────────────────────
    np.random.seed(42)
    model = MiniLM(
        vocab_size  = tok.vocab_size,
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        context_len = CONTEXT_LEN,
    )
    print(f"\n{model}\n")

    # ── 5. אימון ────────────────────────────────────────────────
    print(f"{'Epoch':>6}  {'Loss':>8}  {'זמן':>6}  {'דוגמה'}")
    print("-" * 56)

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ערבוב דוגמאות בכל epoch (SGD אקראי)
        np.random.shuffle(samples)

        epoch_loss = 0.0
        for ctx, target in samples:
            probs      = model.forward(ctx)
            loss       = model.cross_entropy_loss(probs, target)
            grads      = model.backward(target)
            model.update(grads, lr=LR)
            epoch_loss += loss

        avg_loss = epoch_loss / len(samples)
        loss_history.append(avg_loss)
        elapsed  = time.time() - t0

        # הדפסת התקדמות
        if epoch % LOG_EVERY == 0 or epoch == 1:
            seed   = "User: שלום Model:"
            sample = generate(model, tok, seed, n_words=8, temperature=0.7)
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {elapsed:>5.1f}s  \"{sample}\"")

    # ── 6. שמירת מודל ───────────────────────────────────────────
    model_path = f"{SAVE_DIR}/model.npz"
    np.savez(
        model_path,
        E=model.E, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
        config=np.array([
            tok.vocab_size, EMBED_DIM, HIDDEN_DIM, CONTEXT_LEN
        ])
    )
    print(f"\n✅ מודל נשמר → {model_path}")
    print(f"📉 Loss התחלתי: {loss_history[0]:.4f}  →  סופי: {loss_history[-1]:.4f}")

    # ── 7. כמה דוגמאות סיום ─────────────────────────────────────
    print("\n" + "=" * 56)
    print("  🎤  דוגמאות יצירה אחרי אימון")
    print("=" * 56)
    seeds = [
        "User: שלום Model:",
        "User: Hello Model:",
        "User: מה זה ביולוגיה Model:",
        "User: what is mathematics Model:",
        "User: מה זה רשת נוירונים Model:",
    ]
    for seed in seeds:
        out = generate(model, tok, seed, n_words=10, temperature=0.8)
        print(f"  {seed:>8} →  {out}")

    return model, tok, loss_history


# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
