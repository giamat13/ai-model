"""
train.py — לולאת אימון מלאה עם זיהוי שינויים

אימון מצטבר:
  • בכל הרצה מחשבים hash לכל article בנתונים.
  • article שכבר אומן ולא השתנה — מדלג עליו.
  • article חדש או ששונה — מאמן עליו בלבד.
  • אם אין שינויים כלל — לא מאמן בכתב (0 אפוקים), פשוט ממשיך.
  • המצב נשמר בקובץ trained_hashes.json לצד המודל.
"""

import hashlib
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
#  הגדרות אימון
# ════════════════════════════════════════════════════════════════

_HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(_HERE, "training_data.json")
API_DATA_PATH   = os.path.join(_HERE, "api_training_data.json")
FETCHED_PATH    = os.path.join(_HERE, "fetched_articles.json")
MATH_DATA_PATH  = os.path.join(_HERE, "math_training_data.json")
NEW_EXAMPLES_PATH = os.path.join(_HERE, "new_training_examples.json")
NAMING_DATA_PATH = os.path.join(_HERE, "naming_training_data.json")
SOURCES_DIR     = os.path.join(_HERE, "sources")
DATA_PATHS      = [DATA_PATH, API_DATA_PATH, FETCHED_PATH, MATH_DATA_PATH, NEW_EXAMPLES_PATH, NAMING_DATA_PATH]
SAVE_DIR        = _HERE
HASHES_PATH     = os.path.join(SAVE_DIR, "trained_hashes.json")

CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "16"))
EMBED_DIM   = int(os.environ.get("EMBED_DIM",   "64"))
HIDDEN_DIM  = int(os.environ.get("HIDDEN_DIM",  "128"))
EPOCHS      = int(os.environ.get("EPOCHS",      "40"))
LR          = float(os.environ.get("LR",        "0.01"))
LOG_EVERY   = int(os.environ.get("LOG_EVERY",   "10"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "256"))


# ════════════════════════════════════════════════════════════════
#  ניהול Hashes — מה כבר אומן
# ════════════════════════════════════════════════════════════════

def article_hash(article: dict) -> str:
    """hash יחיד לכל article — מבוסס על id + תוכן הטקסט."""
    raw = f"{article.get('id', '')}|{article.get('text', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_trained_hashes() -> dict[str, str]:
    """טוען את המילון {article_id → hash} שנשמר בסוף האימון האחרון."""
    if not os.path.exists(HASHES_PATH):
        return {}
    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_trained_hashes(hashes: dict[str, str]) -> None:
    with open(HASHES_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


def find_changed_articles(
    all_articles: list[dict],
    trained: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """
    מחזיר (changed, unchanged).
    changed   = חדש לגמרי, או ה-text/id שלו השתנה.
    unchanged = hash זהה לזה שנשמר → לא צריך לאמן.
    """
    changed, unchanged = [], []
    for art in all_articles:
        aid  = art.get("id", "")
        h    = article_hash(art)
        if trained.get(aid) == h:
            unchanged.append(art)
        else:
            changed.append(art)
    return changed, unchanged


# ════════════════════════════════════════════════════════════════
#  טעינת נתונים
# ════════════════════════════════════════════════════════════════

def load_all_articles(paths: list[str]) -> list[dict]:
    articles = []
    # טעינת JSON
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        batch = [a for a in data.get("articles", []) if a.get("text")]
        articles.extend(batch)
        print(f"[Data] נטען {os.path.basename(path)}: {len(batch)} פריטים")
    
    # טעינת TXT מתקיית sources
    if os.path.exists(SOURCES_DIR):
        txt_files = [f for f in os.listdir(SOURCES_DIR) if f.endswith(".txt")]
        for txt_file in txt_files:
            txt_path = os.path.join(SOURCES_DIR, txt_file)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                articles.append({
                    "id": f"txt_source_{txt_file}",
                    "text": text,
                    "source": "txt_file",
                    "topic": txt_file
                })
        print(f"[Data] נטענו {len(txt_files)} קבצי TXT מתקיית sources")
    
    if not articles:
        raise FileNotFoundError("לא נמצאו קבצי דאטה לאימון")
    return articles


# ════════════════════════════════════════════════════════════════
#  הכנת דוגמאות אימון
# ════════════════════════════════════════════════════════════════

def make_samples(
    tok: Tokenizer, texts: list[str], context_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    בונה את כל דוגמאות האימון כמערכי NumPy חבילתיים:
      X — (N, context_len) אינדקסי ההקשר
      y — (N,)             אינדקס המילה הבאה לכל דוגמה
    צורת מערך (ולא רשימת tuples) מאפשרת חיתוך מנות מהיר בלולאת האימון.
    """
    ctx_rows: list[list[int]] = []
    targets:  list[int]       = []
    pad_id = tok.word2idx["<PAD>"]
    for text in texts:
        ids = tok.encode(text, add_special=True)
        # חלון שמאל-מרופד לכל טוקן-יעד: מנבאים כל מילה מ-context_len המילים
        # שלפניה, וכשאין מספיק — מרפדים ב-<PAD> משמאל. כך גם טקסטים קצרים
        # (רוב דוגמאות הדו-שיח) תורמים דוגמאות, ומצב האימון תואם לחיזוי
        # ב-chat.py/generate ששם גם מרפדים משמאל.
        for t in range(1, len(ids)):
            target = ids[t]
            if target == pad_id:
                continue
            window = ids[max(0, t - context_len):t]
            if len(window) < context_len:
                window = [pad_id] * (context_len - len(window)) + window
            ctx_rows.append(window)
            targets.append(target)

    if not targets:
        return np.empty((0, context_len), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.array(ctx_rows, dtype=np.int64), np.array(targets, dtype=np.int64)


# ════════════════════════════════════════════════════════════════
#  יצירת טקסט — לבדיקה תוך-כדי אימון
# ════════════════════════════════════════════════════════════════

def generate(
    model: MiniLM, tok: Tokenizer,
    seed_text: str, n_words: int = 12, temperature: float = 0.8,
) -> str:
    pad_id = tok.word2idx["<PAD>"]
    bos_id = tok.word2idx["<BOS>"]
    eos_id = tok.word2idx["<EOS>"]

    base_ids = tok.encode(seed_text, add_special=False) or [bos_id]
    ctx = base_ids[-model.context_len:]
    while len(ctx) < model.context_len:
        ctx = [pad_id] + ctx

    generated = list(base_ids)
    for _ in range(n_words):
        probs   = model.forward(np.array(ctx))
        logits  = np.log(probs + 1e-9) / temperature
        logits -= logits.max()
        probs   = np.exp(logits); probs /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ctx = ctx[1:] + [next_id]

    return tok.decode(generated, skip_special=True)


# ════════════════════════════════════════════════════════════════
#  טעינת מודל קיים (להמשך אימון)
# ════════════════════════════════════════════════════════════════

def load_existing_model(tok: Tokenizer) -> MiniLM | None:
    model_path = os.path.join(SAVE_DIR, "model.npz")
    if not os.path.exists(model_path):
        return None
    model = MiniLM.load(model_path)      # None אם פורמט ישן (ללא Attention)
    if model is None:
        print("[Train] model.npz בפורמט ישן (ללא Attention) — אימון מחדש.")
        return None
    # אם ה-vocab גדל (נוספו מילים) — אי אפשר לטעון, צריך אימון מחדש
    if model.vocab_size != tok.vocab_size:
        print(f"[Train] vocab השתנה ({model.vocab_size} → {tok.vocab_size}) — אימון מחדש.")
        return None
    return model


# ════════════════════════════════════════════════════════════════
#  שמירת מודל
# ════════════════════════════════════════════════════════════════

def save_model(model: MiniLM, tok: Tokenizer) -> None:
    model_path = os.path.join(SAVE_DIR, "model.npz")
    model.save(model_path)
    print(f"[Train] מודל נשמר → {model_path}")


# ════════════════════════════════════════════════════════════════
#  לולאת אימון ראשית
# ════════════════════════════════════════════════════════════════

def train() -> tuple | None:
    print("=" * 56)
    print("  🧠  אימון מצטבר — מאמן רק שינויים")
    print("=" * 56)

    # ── 1. טעינת כל הנתונים ─────────────────────────────────────
    all_articles = load_all_articles(DATA_PATHS)
    trained      = load_trained_hashes()

    # ── 2. זיהוי מה השתנה ──────────────────────────────────────
    changed, unchanged = find_changed_articles(all_articles, trained)
    print(f"[Delta] ללא שינוי: {len(unchanged)}  |  חדש/שונה: {len(changed)}")

    if not changed:
        print("[Train] אין שינויים — מדלג על אימון. ✅")
        # מחזירים None כסימן שלא אומן דבר
        return None

    # ── 3. Tokenizer — תמיד נבנה מכל הנתונים (לשמור vocab עקבי) ─
    tok = Tokenizer()
    tok.build_vocab([a["text"] for a in all_articles], min_freq=1)
    tok.save(os.path.join(SAVE_DIR, "tokenizer.json"))

    # ── 4. מודל — טוען קיים או יוצר חדש ────────────────────────
    np.random.seed(42)
    model = load_existing_model(tok)
    fresh = model is None
    if fresh:
        print("[Train] יוצר מודל חדש.")
        model = MiniLM(
            vocab_size  = tok.vocab_size,
            embed_dim   = EMBED_DIM,
            hidden_dim  = HIDDEN_DIM,
            context_len = CONTEXT_LEN,
        )
    else:
        print(f"[Train] ממשיך אימון על מודל קיים ({model.num_params():,} פרמטרים).")

    print(f"\n{model}\n")

    # ── 5. דוגמאות ─────────────────────────────────────────────
    #   מודל חדש (או שה-vocab גדל → אתחול מאפס) אין לו ידע קודם לשמר,
    #   ולכן חייבים לאמן על כל הקורפוס — אחרת הוא "ישכח" את כל השאר ויכיר
    #   רק את הפריטים החדשים. מודל קיים → אימון מצטבר על מה שהשתנה בלבד.
    train_articles = all_articles if fresh else changed
    train_texts    = [a["text"] for a in train_articles]
    X, y = make_samples(tok, train_texts, CONTEXT_LEN)
    n_samples = len(y)
    if n_samples == 0:
        print("[Train] לא נוצרו דוגמאות לאימון.")
        return None

    scope = "כל הקורפוס (מודל חדש)" if fresh else "מהשינויים בלבד"
    print(f"[Data] דוגמאות אימון ({scope}): {n_samples:,}")

    # ── 6. לולאת אימון חבילתית (mini-batch) ────────────────────
    #   מעבדים BATCH_SIZE דוגמאות בכל צעד בכפל-מטריצות אחד (BLAS),
    #   במקום דוגמה-בכל-פעם. הגרדיאנטים ממוצעים על המנה, ולכן ה-LR
    #   מוגדל ליניארית (linear scaling rule) כדי לשמר את גודל הצעד
    #   האפקטיבי לכל אפוק — אותה התכנסות, זמן-קיר קצר בהרבה.
    batch_size = min(BATCH_SIZE, n_samples)
    eff_lr     = LR * batch_size
    n_batches  = (n_samples + batch_size - 1) // batch_size
    print(f"[Train] מנה={batch_size}  |  מנות/אפוק={n_batches:,}  |  LR אפקטיבי={eff_lr:.4f}")

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'זמן':>6}  {'דוגמה'}")
    print("-" * 56)

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        perm = np.random.permutation(n_samples)
        Xs, ys = X[perm], y[perm]

        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size]
            probs = model.forward_batch(xb)
            loss  = model.cross_entropy_loss_batch(probs, yb)
            grads = model.backward_batch(yb)
            model.update(grads, lr=eff_lr)
            epoch_loss += loss * len(yb)          # loss ממוצע-מנה → משוקלל לפי גודל

        avg_loss = epoch_loss / n_samples
        loss_history.append(avg_loss)
        elapsed  = time.time() - t0

        if epoch % LOG_EVERY == 0 or epoch == 1:
            seed   = "User: שלום Model:"
            sample = generate(model, tok, seed, n_words=8, temperature=0.7)
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {elapsed:>5.1f}s  \"{sample}\"")

    # ── 7. שמירה ─────────────────────────────────────────────────
    save_model(model, tok)

    # עדכן hashes — כל מה שנמצא כרגע (כולל unchanged) מעודכן
    new_hashes = {a["id"]: article_hash(a) for a in all_articles if a.get("id")}
    save_trained_hashes(new_hashes)
    print(f"[Train] hashes נשמרו ({len(new_hashes)} articles).")

    print(f"\n✅ Loss ראשוני: {loss_history[0]:.4f}  →  סופי: {loss_history[-1]:.4f}")

    # ── 8. דוגמאות סיום ─────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  🎤  דוגמאות יצירה אחרי אימון")
    print("=" * 56)
    for seed in [
        "User: שלום Model:",
        "User: Hello Model:",
        "User: מה זה ביולוגיה Model:",
        "User: what is mathematics Model:",
    ]:
        out = generate(model, tok, seed, n_words=10, temperature=0.8)
        print(f"  {seed} →  {out}")

    return model, tok, loss_history


# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
