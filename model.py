"""
model.py — רשת נוירונים מאפס, NumPy בלבד

ארכיטקטורה:
  [מילה_קלט] → Embedding → Linear → ReLU → Linear → Softmax → [הסתברויות]

  • Embedding:  כל מילה הופכת לוקטור צף (embed_dim מימדים)
  • Linear:     כפל מטריצות + ביאס  (y = xW + b)
  • ReLU:       max(0, x)  — מוסיף אי-לינאריות
  • Softmax:    הופך את הפלט להסתברויות המסתכמות ל-1

המודל מקבל חלון של n מילים אחרונות ומנבא את המילה הבאה.
"""

import numpy as np


# ── פונקציות אקטיבציה ──────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: מחזיר 0 לערכים שליליים, x לחיוביים"""
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    """נגזרת של ReLU: 1 איפה x>0, אחרת 0"""
    return (x > 0).astype(float)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax יציב מספרית.
    מחסירים את המקסימום לפני exp כדי למנוע overflow.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# ── המודל הראשי ─────────────────────────────────────────────────────────────

class MiniLM:
    """
    מודל שפה זעיר (Language Model).

    פרמטרים:
      vocab_size  — גודל אוצר המילים
      embed_dim   — מימד הייצוג של כל מילה (למשל 64)
      hidden_dim  — מספר הנוירונים בשכבה הנסתרת
      context_len — כמה מילים אחורה המודל מסתכל
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        hidden_dim:  int = 128,
        context_len: int = 4,
    ):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.context_len = context_len

        # ── אתחול משקולות (He initialization) ──────────────────────────────
        # E  — מטריצת ה-Embedding: שורה אחת לכל מילה
        self.E = np.random.randn(vocab_size, embed_dim) * 0.01

        # W1, b1 — שכבה ראשונה: הכניסה היא שרשור של context_len וקטורי embedding
        input_dim = context_len * embed_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros(hidden_dim)

        # W2, b2 — שכבה שנייה: הפלט הוא הסתברות על כל המילים
        self.W2 = np.random.randn(hidden_dim, vocab_size) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(vocab_size)

        # ── cache לפעפוע לאחור ──────────────────────────────────────────────
        self._cache: dict = {}

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, context_ids: np.ndarray) -> np.ndarray:
        """
        context_ids: מערך של context_len מספרים שלמים (אינדקסים של מילים)
        מחזיר: וקטור הסתברויות באורך vocab_size
        """
        # 1. Embedding lookup — שליפת וקטור לכל מילה בהקשר
        emb = self.E[context_ids]           # (context_len, embed_dim)
        x   = emb.flatten()                # (context_len * embed_dim,)

        # 2. שכבה נסתרת
        h_pre = x @ self.W1 + self.b1     # (hidden_dim,)
        h     = relu(h_pre)               # (hidden_dim,)  ← אי-לינאריות

        # 3. שכבת פלט
        logits = h @ self.W2 + self.b2    # (vocab_size,)
        probs  = softmax(logits)          # (vocab_size,)  ← הסתברויות

        # שמירת ערכי ביניים לשלב backprop
        self._cache = {
            "context_ids": context_ids,
            "x": x, "h_pre": h_pre, "h": h,
            "logits": logits, "probs": probs,
        }
        return probs

    # ── Loss ────────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, target_id: int) -> float:
        """
        Cross-Entropy Loss: -log(p[target])
        מודד כמה המודל בטוח בתשובה הנכונה.
        ככל שהמודל מאמין יותר בתשובה הנכונה, ה-loss קטן יותר.
        """
        p = probs[target_id]
        return -np.log(p + 1e-9)   # +1e-9 כדי למנוע log(0)

    # ── Backward pass (Backpropagation) ─────────────────────────────────────

    def backward(self, target_id: int) -> dict:
        """
        מחשב גרדיאנטים של כל הפרמטרים ביחס ל-Loss.
        
        הגרדיאנט אומר: "אם אגדיל פרמטר זה ב-ε, ה-Loss ישתנה ב-gradient*ε"
        לאחר מכן נעדכן כל פרמטר בכיוון ההפוך (ירידת גרדיאנט).
        """
        c = self._cache

        # ── גרדיאנט של Loss לפי logits ──────────────────────────────────────
        # כשמשתמשים ב-cross-entropy + softmax, הנגזרת היא פשוט:
        #   dL/d(logits[i]) = probs[i] - 1(i==target)
        dlogits = c["probs"].copy()
        dlogits[target_id] -= 1.0      # (vocab_size,)

        # ── שכבה שנייה ──────────────────────────────────────────────────────
        dW2 = np.outer(c["h"], dlogits)        # (hidden_dim, vocab_size)
        db2 = dlogits                           # (vocab_size,)
        dh  = self.W2 @ dlogits                # (hidden_dim,)

        # ── נגזרת דרך ReLU ───────────────────────────────────────────────────
        dh_pre = dh * relu_grad(c["h_pre"])    # (hidden_dim,)

        # ── שכבה ראשונה ──────────────────────────────────────────────────────
        dW1 = np.outer(c["x"], dh_pre)         # (input_dim, hidden_dim)
        db1 = dh_pre                            # (hidden_dim,)
        dx  = self.W1 @ dh_pre                 # (input_dim,)

        # ── Embedding ────────────────────────────────────────────────────────
        dx_reshaped = dx.reshape(self.context_len, self.embed_dim)
        dE = np.zeros_like(self.E)
        for pos, idx in enumerate(c["context_ids"]):
            dE[idx] += dx_reshaped[pos]        # מצטבר אם אותה מילה מופיעה כמה פעמים

        return {"dE": dE, "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ── עדכון משקולות ───────────────────────────────────────────────────────

    def update(self, grads: dict, lr: float) -> None:
        """
        SGD פשוט: param -= lr * gradient
        lr (learning rate) קובע את גודל הצעד — גדול מדי = אי-יציבות, קטן מדי = למידה איטית
        """
        self.E  -= lr * grads["dE"]
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    # ── מידע ────────────────────────────────────────────────────────────────

    def num_params(self) -> int:
        """סך הפרמטרים במודל"""
        return (
            self.E.size + self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )

    def __repr__(self) -> str:
        return (
            f"MiniLM(\n"
            f"  vocab={self.vocab_size}, embed={self.embed_dim}, "
            f"hidden={self.hidden_dim}, ctx={self.context_len}\n"
            f"  פרמטרים: {self.num_params():,}\n"
            f")"
        )


# ── בדיקה עצמאית ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    model = MiniLM(vocab_size=500, embed_dim=32, hidden_dim=64, context_len=3)
    print(model)

    # forward
    ctx = np.array([1, 5, 12])
    probs = model.forward(ctx)
    print(f"\nforward — סכום הסתברויות: {probs.sum():.6f}  (חייב להיות 1.0)")
    print(f"הסתברות מקסימלית: {probs.max():.4f}  (מילה {probs.argmax()})")

    # loss + backward
    target = 42
    loss = model.cross_entropy_loss(probs, target)
    print(f"\nloss ראשוני: {loss:.4f}  (צפוי ≈ log({model.vocab_size}) = {np.log(model.vocab_size):.2f})")

    grads = model.backward(target)
    model.update(grads, lr=0.01)
    print("\n✅ backward + update עברו בהצלחה")
