"""
tokenizer.py — ממיר מילים למספרים

למה זה נחוץ?
  רשת נוירונים עובדת עם מספרים, לא עם מילים.
  כל מילה ייחודית מקבלת מספר שלם (index).
  לדוגמה: "בינה"→4, "מלאכותית"→7, "היא"→2 ...
"""

import json
import re
from collections import Counter


# ── טוקנים מיוחדים ──────────────────────────────────────────────────────────
PAD   = "<PAD>"    # ריפוד — ממלא רצפים קצרים כדי שיהיו באורך אחיד
UNK   = "<UNK>"    # מילה לא מוכרת (out-of-vocabulary)
BOS   = "<BOS>"    # Beginning Of Sentence — תחילת משפט
EOS   = "<EOS>"    # End Of Sentence — סוף משפט


class Tokenizer:
    """
    אחראי על:
      1. בניית אוצר מילים (vocab) מהטקסט
      2. המרה: מילה → מספר  (encode)
      3. המרה: מספר → מילה  (decode)
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}   # מילה → מספר
        self.idx2word: dict[int, str] = {}   # מספר → מילה
        self.vocab_size: int = 0

    # ── עיבוד טקסט גולמי ────────────────────────────────────────────────────

    @staticmethod
    def clean(text: str) -> list[str]:
        """
        מנקה טקסט ומפצל למילים.
        שלבים:
          • מוריד סימני פיסוק מסביב למילים (אך שומר על הסדר)
          • מוריד תווים לא עבריים/לועזיים מיותרים
          • מפצל לפי רווחים
        """
        # נרמול: בגד-כפת עם ניקוד → הסר ניקוד (אם יש)
        text = re.sub(r'[\u0591-\u05C7]', '', text)   # ניקוד עברי
        # החלף סימני פיסוק ברווח
        text = re.sub(r'[–—״׳"\'«»\(\)\[\]{}]', ' ', text)
        # שמור נקודה/פסיק/נקודותיים כטוקן נפרד
        text = re.sub(r'([.,;:!?])', r' \1 ', text)
        # כמה רווחים → רווח אחד
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower().split()

    # ── בניית אוצר מילים ────────────────────────────────────────────────────

    def build_vocab(self, texts: list[str], min_freq: int = 1) -> None:
        """
        סורק את כל הטקסטים ובונה vocab.

        min_freq: מילים שמופיעות פחות מ-min_freq פעמים יוחלפו ב-<UNK>.
                  בקורפוס קטן שלנו נשתמש ב-1 (כל מילה נכנסת).
        """
        # ספירת כל המילים
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.clean(text))

        # טוקנים מיוחדים תמיד ראשונים (אינדקסים קבועים 0-3)
        special = [PAD, UNK, BOS, EOS]
        vocab = special + [w for w, c in counter.most_common() if c >= min_freq]

        # בניית מילונים דו-כיווניים
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"[Tokenizer] אוצר מילים: {self.vocab_size:,} מילים")
        print(f"[Tokenizer] טוקנים מיוחדים: {special}")

    # ── קידוד ופענוח ────────────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """מחרוזת → רשימת מספרים שלמים"""
        tokens = self.clean(text)
        ids = [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens]
        if add_special:
            ids = [self.word2idx[BOS]] + ids + [self.word2idx[EOS]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """רשימת מספרים → מחרוזת"""
        special_ids = {self.word2idx[t] for t in [PAD, BOS, EOS]}
        words = []
        for i in ids:
            w = self.idx2word.get(i, UNK)
            if skip_special and i in special_ids:
                continue
            words.append(w)
        return ' '.join(words)

    # ── שמירה וטעינה ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {"word2idx": self.word2idx, "idx2word": self.idx2word}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] נשמר → {path}")

    def load(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] נטען ← {path}  ({self.vocab_size:,} מילים)")


# ── בדיקה עצמאית ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "בינה מלאכותית היא יכולת של מחשבים ללמוד מנתונים.",
        "רשת נוירונים לומדת על ידי עדכון משקולות.",
        "למידה עמוקה משתמשת בשכבות רבות של נוירונים."
    ]

    tok = Tokenizer()
    tok.build_vocab(sample, min_freq=1)

    sentence = "רשת נוירונים לומדת"
    encoded = tok.encode(sentence)
    decoded = tok.decode(encoded)

    print(f"\nמשפט מקורי : '{sentence}'")
    print(f"מקודד      : {encoded}")
    print(f"מפוענח     : '{decoded}'")
