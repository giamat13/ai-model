"""
model.py — רשת נוירונים ב-PyTorch (עם Self-Attention)

ארכיטקטורה (בלוק טרנספורמר מינימלי, ראש-בודד) — זהה לגרסת ה-NumPy הקודמת,
רק שכעת הפעפוע-לאחור נעשה אוטומטית ע"י autograd של PyTorch:

  [מילים_קלט]
     │  Embedding לכל מילה  +  Positional embedding לכל מיקום
     ▼
  x  (B, T, D)
     │  Self-Attention:  Q=xWq, K=xWk, V=xWv
     │                   A = softmax(QKᵀ / √D)
     │                   attn = A·V ,  proj = attn·Wo
     │  Residual:        x2 = x + proj
     ▼
  x2[:, -1]  ← לוקחים רק את הייצוג של המיקום האחרון (הוא ש"רואה" את כל ההקשר)
     │  Linear → ReLU → Linear
     ▼
  [logits על המילה הבאה]

הערה על מסכה סיבתית (causal mask):
  אנחנו משתמשים *רק* בייצוג של המיקום האחרון לחיזוי המילה הבאה. המיקום
  האחרון ממילא רשאי להסתכל על כל המיקומים הקודמים, ולכן מסכה סיבתית ומסכה
  מלאה זהות עבור הפלט שלנו — אין צורך במסכה.

הערה על מעבר מ-NumPy ל-PyTorch:
  קודם כתבנו forward/backward/update ידנית. עכשיו forward בלבד מספיק —
  autograd גוזר את כל הגרדיאנטים, ו-optimizer סטנדרטי (SGD) מעדכן. גזירת-נורמה
  (gradient clipping) עדיין נחוצה כדי לשמור על יציבות ה-Attention, אבל היא
  נעשית ב-train.py דרך torch.nn.utils.clip_grad_norm_.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniLM(nn.Module):
    """
    מודל שפה זעיר עם Self-Attention (ראש בודד).

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
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.context_len = context_len

        D = embed_dim
        T = context_len

        # ── Embeddings ──────────────────────────────────────────────────────
        # שורה אחת לכל מילה, ווקטור נלמד לכל מיקום בחלון (0..T-1)
        self.tok_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Parameter(torch.randn(T, D) * 0.01)

        # ── Self-Attention (ראש בודד) — השלכות Q/K/V/O, כל אחת (D, D) ─────────
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)

        # ── ראש Feed-Forward (מנבא את המילה הבאה מהמיקום האחרון) ────────────
        self.fc1 = nn.Linear(D, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """אתחול צנוע — שומר על סקאלה קטנה כמו בגרסת ה-NumPy כדי למנוע
        התפוצצות ב-Attention בתחילת האימון."""
        nn.init.normal_(self.tok_emb.weight, std=0.01)
        attn_scale = 1.0 / math.sqrt(self.embed_dim)
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.normal_(lin.weight, std=attn_scale)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        """
        context_ids: (B, T) אינדקסי מילים (LongTensor)
        מחזיר:       (B, vocab_size) logits (לא softmax — CrossEntropyLoss עושה זאת)
        """
        D = self.embed_dim
        scale = 1.0 / math.sqrt(D)

        # 1. Embedding של מילים + מיקום  → x (B, T, D)
        x = self.tok_emb(context_ids) + self.pos_emb.unsqueeze(0)

        # 2. השלכות Q, K, V  → כל אחת (B, T, D)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        # 3. ציוני תשומת-לב:  QKᵀ / √D  → (B, T, T) ואז softmax על המפתחות
        scores = torch.matmul(Q, K.transpose(1, 2)) * scale
        A = F.softmax(scores, dim=-1)

        # 4. שקלול הערכים + השלכת פלט
        attn = torch.matmul(A, V)          # (B, T, D)
        proj = self.Wo(attn)               # (B, T, D)

        # 5. חיבור שיורי (residual)
        x2 = x + proj                      # (B, T, D)

        # 6. לוקחים את המיקום האחרון בלבד → ראש Feed-Forward
        last = x2[:, -1, :]                # (B, D)
        h = F.relu(self.fc1(last))         # (B, hidden)
        logits = self.fc2(h)               # (B, vocab)
        return logits

    def forward_one(self, context_ids: torch.Tensor) -> torch.Tensor:
        """גרסת דוגמה-בודדת — מקבלת (T,) ומחזירה logits בצורת (vocab_size,)."""
        return self.forward(context_ids.unsqueeze(0))[0]

    # ── שמירה וטעינה ────────────────────────────────────────────────────────
    #   מקור-אמת יחיד לפורמט model.pt — train.py ו-chat.py משתמשים בו במקום
    #   לשכפל את לוגיקת הטעינה.

    def config(self) -> dict:
        return {
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "hidden_dim":  self.hidden_dim,
            "context_len": self.context_len,
        }

    def save(self, path: str) -> None:
        torch.save({"config": self.config(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "MiniLM | None":
        """
        טוען מודל מ-model.pt. מחזיר None אם הקובץ בפורמט לא-תואם
        (למשל model.npz ישן מגרסת ה-NumPy) — הקורא יחליט לאמן מחדש.
        """
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            return None
        if not isinstance(ckpt, dict) or "config" not in ckpt or "state_dict" not in ckpt:
            return None
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    # ── מידע ────────────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
    torch.manual_seed(42)

    model = MiniLM(vocab_size=500, embed_dim=32, hidden_dim=64, context_len=6)
    print(model)

    # forward — דוגמה בודדת
    ctx = torch.tensor([1, 5, 12, 7, 3, 9])
    logits = model.forward_one(ctx)
    probs = F.softmax(logits, dim=-1)
    print(f"\nforward — סכום הסתברויות: {probs.sum().item():.6f}  (חייב להיות 1.0)")
    print(f"הסתברות מקסימלית: {probs.max().item():.4f}  (מילה {probs.argmax().item()})")

    target = torch.tensor([42])
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    print(f"\nloss ראשוני: {loss.item():.4f}  "
          f"(צפוי ≈ log({model.vocab_size}) = {math.log(model.vocab_size):.2f})")

    # ── בדיקה שה-autograd + עדכון מורידים את ה-loss על מנה קטנה ──────────────
    print("\n── בדיקת אימון קצרה (autograd + SGD) ──")
    torch.manual_seed(0)
    m = MiniLM(vocab_size=30, embed_dim=8, hidden_dim=16, context_len=4)
    Xb = torch.randint(0, 30, size=(16, 4))
    yb = torch.randint(0, 30, size=(16,))
    opt = torch.optim.SGD(m.parameters(), lr=0.5)

    first = last = None
    for step in range(50):
        opt.zero_grad()
        loss = F.cross_entropy(m(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"  loss: {first:.4f} → {last:.4f}")
    assert last < first, "האימון לא הוריד את ה-loss!"

    # ── בדיקת save/load round-trip ──────────────────────────────────────────
    import os, tempfile
    tmp = os.path.join(tempfile.gettempdir(), "minilm_test.pt")
    m.save(tmp)
    m2 = MiniLM.load(tmp)
    assert m2 is not None, "load החזיר None!"
    with torch.no_grad():
        assert torch.allclose(m(Xb), m2(Xb)), "המודל הטעון מפיק פלט שונה!"
    os.remove(tmp)

    print("\n✅ כל הבדיקות עברו (forward, אימון, save/load)")
