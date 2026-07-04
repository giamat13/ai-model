"""
model.py — רשת נוירונים מאפס, NumPy בלבד (עם Self-Attention)

ארכיטקטורה (בלוק טרנספורמר מינימלי, ראש-בודד):

  [מילים_קלט]
     │  Embedding לכל מילה  +  Positional embedding לכל מיקום
     ▼
  x  (T, D)
     │  Self-Attention:  Q=xWq, K=xWk, V=xWv
     │                   A = softmax(QKᵀ / √D)
     │                   attn = A·V ,  proj = attn·Wo
     │  Residual:        x2 = x + proj
     ▼
  x2[-1]   ← לוקחים רק את הייצוג של המיקום האחרון (הוא ש"רואה" את כל ההקשר)
     │  Linear → ReLU → Linear → Softmax
     ▼
  [הסתברויות על המילה הבאה]

הרעיון המרכזי לעומת הגרסה הקודמת:
  קודם שרשרנו את כל וקטורי ההקשר (flatten) והזנו ל-W1 יחיד — כל מיקום קיבל
  משקל קבוע משלו, ולא היה "דיבור" בין מילים. עכשיו ה-Attention מאפשר לכל
  מילה להסתכל על כל שאר המילים בחלון ולשקלל אותן דינמית — בדיוק כמו ב-GPT.

הערה על מסכה סיבתית (causal mask):
  אנחנו משתמשים *רק* בייצוג של המיקום האחרון לחיזוי המילה הבאה. המיקום
  האחרון ממילא רשאי להסתכל על כל המיקומים הקודמים, ולכן מסכה סיבתית ומסכה
  מלאה זהות עבור הפלט שלנו — אין צורך במסכה. (המיקומים האחרים מחושבים אך
  נזרקים, והגרדיאנט שלהם 0.)
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
    Softmax יציב מספרית (על הציר האחרון).
    מחסירים את המקסימום לפני exp כדי למנוע overflow.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# ── המודל הראשי ─────────────────────────────────────────────────────────────

class MiniLM:
    """
    מודל שפה זעיר עם Self-Attention.

    פרמטרים:
      vocab_size  — גודל אוצר המילים
      embed_dim   — מימד הייצוג של כל מילה  (D)
      hidden_dim  — מספר הנוירונים בשכבת ה-Feed-Forward
      context_len — כמה מילים אחורה המודל מסתכל  (T)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        hidden_dim:  int = 128,
        context_len: int = 8,
    ):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.context_len = context_len

        D = embed_dim
        T = context_len

        # ── Embeddings ──────────────────────────────────────────────────────
        # E — Embedding של מילים: שורה אחת לכל מילה
        self.E = np.random.randn(vocab_size, D) * 0.01
        # P — Positional embedding: וקטור נלמד לכל מיקום בחלון (0..T-1)
        self.P = np.random.randn(T, D) * 0.01

        # ── Self-Attention (ראש בודד) ───────────────────────────────────────
        # השלכות Query / Key / Value / Output, כל אחת (D, D)
        attn_scale = 1.0 / np.sqrt(D)
        self.Wq = np.random.randn(D, D) * attn_scale
        self.Wk = np.random.randn(D, D) * attn_scale
        self.Wv = np.random.randn(D, D) * attn_scale
        self.Wo = np.random.randn(D, D) * attn_scale

        # ── ראש Feed-Forward (מנבא את המילה הבאה מהמיקום האחרון) ────────────
        # W1: (D → hidden) ,  W2: (hidden → vocab)
        self.W1 = np.random.randn(D, hidden_dim) * np.sqrt(2 / D)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, vocab_size) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(vocab_size)

        # ── cache לפעפוע לאחור ──────────────────────────────────────────────
        self._cache: dict = {}

    # ── Forward pass — מנה שלמה (Batch) ──────────────────────────────────────

    def forward_batch(self, context_ids: np.ndarray) -> np.ndarray:
        """
        מעבד B דוגמאות בבת-אחת דרך בלוק ה-Attention והראש.

        context_ids: (B, T) אינדקסי מילים
        מחזיר:       (B, vocab_size) הסתברויות
        """
        B = context_ids.shape[0]
        T = self.context_len
        D = self.embed_dim
        scale = 1.0 / np.sqrt(D)

        # 1. Embedding של מילים + מיקום  → x (B, T, D)
        tok_emb = self.E[context_ids]              # (B, T, D)
        x       = tok_emb + self.P[None, :, :]     # broadcast P על כל המנה

        # 2. השלכות Q, K, V  → כל אחת (B, T, D)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # 3. ציוני תשומת-לב:  QKᵀ / √D  → (B, T, T)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale
        A      = softmax(scores)                   # (B, T, T)  softmax על מפתחות

        # 4. שקלול הערכים + השלכת פלט
        attn = np.matmul(A, V)                     # (B, T, D)
        proj = attn @ self.Wo                      # (B, T, D)

        # 5. חיבור שיורי (residual)
        x2 = x + proj                              # (B, T, D)

        # 6. לוקחים את המיקום האחרון בלבד → ראש Feed-Forward
        last  = x2[:, -1, :]                        # (B, D)
        h_pre = last @ self.W1 + self.b1           # (B, hidden)
        h     = relu(h_pre)                         # (B, hidden)
        logits = h @ self.W2 + self.b2             # (B, vocab)
        probs  = softmax(logits)                    # (B, vocab)

        self._cache = {
            "context_ids": context_ids, "B": B,
            "x": x, "Q": Q, "K": K, "V": V,
            "A": A, "attn": attn, "last": last,
            "h_pre": h_pre, "h": h, "probs": probs,
        }
        return probs

    def forward(self, context_ids: np.ndarray) -> np.ndarray:
        """גרסת דוגמה-בודדת — עוטפת את forward_batch. מחזירה וקטור (vocab_size,)."""
        context_ids = np.asarray(context_ids)
        return self.forward_batch(context_ids[None, :])[0]

    # ── Loss ────────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, target_id: int) -> float:
        """Cross-Entropy לדוגמה בודדת: -log(p[target])"""
        return float(-np.log(probs[target_id] + 1e-9))

    @staticmethod
    def cross_entropy_loss_batch(probs: np.ndarray, targets: np.ndarray) -> float:
        """Cross-Entropy ממוצע על כל המנה — (B, vocab) + (B,) targets."""
        B = probs.shape[0]
        p = probs[np.arange(B), targets]
        return float(np.mean(-np.log(p + 1e-9)))

    # ── Backward pass — מנה שלמה (Batch) ─────────────────────────────────────

    def backward_batch(self, targets: np.ndarray) -> dict:
        """
        גרדיאנטים *ממוצעים* על כל המנה, דרך הראש, החיבור השיורי וה-Attention.
        targets: (B,) — האינדקס הנכון לכל דוגמה.
        """
        c     = self._cache
        B     = c["B"]
        D     = self.embed_dim
        scale = 1.0 / np.sqrt(D)

        # ── ראש: dL/d(logits) = probs - one_hot, ממוצע על המנה ─────────────
        dlogits = c["probs"].copy()
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B                                    # (B, vocab)

        dW2 = c["h"].T @ dlogits                        # (hidden, vocab)
        db2 = dlogits.sum(axis=0)                       # (vocab,)
        dh  = dlogits @ self.W2.T                       # (B, hidden)

        dh_pre = dh * relu_grad(c["h_pre"])             # (B, hidden)
        dW1 = c["last"].T @ dh_pre                      # (D, hidden)
        db1 = dh_pre.sum(axis=0)                        # (hidden,)
        dlast = dh_pre @ self.W1.T                      # (B, D)

        # ── רק המיקום האחרון של x2 קיבל גרדיאנט ─────────────────────────────
        dx2 = np.zeros_like(c["x"])                     # (B, T, D)
        dx2[:, -1, :] = dlast

        # ── חיבור שיורי:  x2 = x + proj  ────────────────────────────────────
        dproj = dx2                                     # (B, T, D)
        dx    = dx2.copy()                              # החלק שעוקף את ה-Attention

        # ── proj = attn @ Wo ───────────────────────────────────────────────
        dWo   = np.einsum("btd,bte->de", c["attn"], dproj)   # (D, D)
        dattn = dproj @ self.Wo.T                            # (B, T, D)

        # ── attn = A @ V ───────────────────────────────────────────────────
        dA = np.matmul(dattn, c["V"].transpose(0, 2, 1))     # (B, T, T)
        dV = np.matmul(c["A"].transpose(0, 2, 1), dattn)     # (B, T, D)

        # ── A = softmax(scores)  (על הציר האחרון, לכל שורה) ─────────────────
        #   ds = A ⊙ (dA - Σ(dA ⊙ A))
        tmp     = np.sum(dA * c["A"], axis=-1, keepdims=True)
        dscores = c["A"] * (dA - tmp)                        # (B, T, T)

        # ── scores = (Q Kᵀ) · scale ────────────────────────────────────────
        dscores *= scale
        dQ = np.matmul(dscores, c["K"])                      # (B, T, D)
        dK = np.matmul(dscores.transpose(0, 2, 1), c["Q"])   # (B, T, D)

        # ── Q,K,V = x @ Wq,Wk,Wv ───────────────────────────────────────────
        dWq = np.einsum("btd,bte->de", c["x"], dQ)           # (D, D)
        dWk = np.einsum("btd,bte->de", c["x"], dK)
        dWv = np.einsum("btd,bte->de", c["x"], dV)
        dx += dQ @ self.Wq.T
        dx += dK @ self.Wk.T
        dx += dV @ self.Wv.T                                 # (B, T, D)

        # ── x = tok_emb + P ────────────────────────────────────────────────
        dP = dx.sum(axis=0)                                  # (T, D)
        dE = np.zeros_like(self.E)
        np.add.at(
            dE,
            c["context_ids"].reshape(-1),                    # (B*T,)
            dx.reshape(-1, D),
        )

        return {
            "dE": dE, "dP": dP,
            "dWq": dWq, "dWk": dWk, "dWv": dWv, "dWo": dWo,
            "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
        }

    def backward(self, target_id: int) -> dict:
        """גרסת דוגמה-בודדת — עוטפת את backward_batch."""
        return self.backward_batch(np.array([target_id]))

    # ── עדכון משקולות ───────────────────────────────────────────────────────

    def update(self, grads: dict, lr: float, max_norm: float | None = 1.0) -> None:
        """
        SGD עם גזירת-נורמה (gradient clipping).

        ב-Attention מכפלות QKᵀ הופכות את משטח ה-Loss לרגיש הרבה יותר, וצעד
        גדול מדי גורם ל-overflow → NaN. גזירת הנורמה הגלובלית של הגרדיאנטים
        לתקרה max_norm שומרת על יציבות בלי לשנות את *כיוון* הצעד — טכניקת
        אימון סטנדרטית לטרנספורמרים (לא אלגוריתם ייעודי, היגיינת-אימון).
        """
        if max_norm is not None:
            total = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
            if total > max_norm:
                scale = max_norm / (total + 1e-9)
                grads = {k: v * scale for k, v in grads.items()}

        self.E  -= lr * grads["dE"]
        self.P  -= lr * grads["dP"]
        self.Wq -= lr * grads["dWq"]
        self.Wk -= lr * grads["dWk"]
        self.Wv -= lr * grads["dWv"]
        self.Wo -= lr * grads["dWo"]
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    # ── שמירה וטעינה ────────────────────────────────────────────────────────
    #   מקור-אמת יחיד לפורמט model.npz — train.py ו-chat.py משתמשים בו במקום
    #   לשכפל את לוגיקת הטעינה (וכך משקולות חדשות לא "נשכחות" בשקט).

    def save(self, path: str) -> None:
        np.savez(
            path,
            E=self.E, P=self.P,
            Wq=self.Wq, Wk=self.Wk, Wv=self.Wv, Wo=self.Wo,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            config=np.array([self.vocab_size, self.embed_dim,
                             self.hidden_dim, self.context_len]),
        )

    @classmethod
    def load(cls, path: str) -> "MiniLM | None":
        """
        טוען מודל מ-model.npz. מחזיר None אם הקובץ בפורמט ישן
        (מלפני שדרוג ה-Attention) — הקורא יחליט לאמן מחדש.
        """
        data   = np.load(path, allow_pickle=True)
        config = data["config"]
        if "Wq" not in data.files:
            return None
        model = cls(int(config[0]), int(config[1]), int(config[2]), int(config[3]))
        model.E  = data["E"];  model.P  = data["P"]
        model.Wq = data["Wq"]; model.Wk = data["Wk"]
        model.Wv = data["Wv"]; model.Wo = data["Wo"]
        model.W1 = data["W1"]; model.b1 = data["b1"]
        model.W2 = data["W2"]; model.b2 = data["b2"]
        return model

    # ── מידע ────────────────────────────────────────────────────────────────

    def num_params(self) -> int:
        """סך הפרמטרים במודל"""
        return (
            self.E.size + self.P.size +
            self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size +
            self.W1.size + self.b1.size + self.W2.size + self.b2.size
        )

    def __repr__(self) -> str:
        return (
            f"MiniLM+Attention(\n"
            f"  vocab={self.vocab_size}, embed={self.embed_dim}, "
            f"hidden={self.hidden_dim}, ctx={self.context_len}\n"
            f"  פרמטרים: {self.num_params():,}\n"
            f")"
        )


# ── בדיקה עצמאית ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    model = MiniLM(vocab_size=500, embed_dim=32, hidden_dim=64, context_len=6)
    print(model)

    # forward — דוגמה בודדת
    ctx = np.array([1, 5, 12, 7, 3, 9])
    probs = model.forward(ctx)
    print(f"\nforward — סכום הסתברויות: {probs.sum():.6f}  (חייב להיות 1.0)")
    print(f"הסתברות מקסימלית: {probs.max():.4f}  (מילה {probs.argmax()})")

    target = 42
    loss = model.cross_entropy_loss(probs, target)
    print(f"\nloss ראשוני: {loss:.4f}  (צפוי ≈ log({model.vocab_size}) = {np.log(model.vocab_size):.2f})")

    grads = model.backward(target)
    model.update(grads, lr=0.01)
    print("✅ backward + update (דוגמה בודדת) עברו")

    # ── בדיקת גרדיאנט נומרית — מוודאת שה-backprop של ה-Attention נכון ──────
    print("\n── בדיקת גרדיאנט נומרית ──")
    np.random.seed(0)
    m = MiniLM(vocab_size=30, embed_dim=8, hidden_dim=16, context_len=4)
    Xb = np.random.randint(0, 30, size=(3, 4))
    yb = np.random.randint(0, 30, size=(3,))

    def loss_of(model):
        p = model.forward_batch(Xb)
        return model.cross_entropy_loss_batch(p, yb)

    m.forward_batch(Xb)
    analytic = m.backward_batch(yb)

    eps = 1e-5
    max_err = 0.0
    for name, pname in [("dWq", "Wq"), ("dWo", "Wo"), ("dP", "P"),
                        ("dW1", "W1"), ("dW2", "W2")]:
        param = getattr(m, pname)
        num = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = param[i]
            param[i] = orig + eps; lp = loss_of(m)
            param[i] = orig - eps; lm = loss_of(m)
            param[i] = orig
            num[i] = (lp - lm) / (2 * eps)
            it.iternext()
        err = np.max(np.abs(num - analytic[name]))
        max_err = max(max_err, err)
        print(f"  {pname:>4}: שגיאה מקס' {err:.2e}")

    assert max_err < 1e-4, f"בדיקת גרדיאנט נכשלה! שגיאה {max_err:.2e}"
    print(f"\n✅ בדיקת גרדיאנט עברה (שגיאה מקס' {max_err:.2e})")
