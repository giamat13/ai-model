"""
run.py — מאמן אם יש שינויים, ואז פותח את הצ'אט.

בכל הרצה:
  1. train() רץ — אם אין שינויים בנתונים הוא מדלג מיידית (אפס עלות זמן).
  2. אם יש שינויים — מאמן רק עליהם, מעדכן המודל הקיים.
  3. פותח GUI (דרך chat.py — מקור יחיד לכל לוגיקת התצוגה).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.npz")
TOK_PATH   = os.path.join(_HERE, "tokenizer.json")


def has_base_model() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(TOK_PATH)


def run_train():
    """
    תמיד קורא ל-train().
    אם אין שינויים בנתונים — train() מחזיר None ומסיים תוך שנייה.
    אם יש שינויים — מאמן רק אותם ומעדכן model.npz.
    """
    from train import train
    result = train()
    if result is None and not has_base_model():
        print("[Run] לא נמצא מודל ואין שינויים לאמן — בודק שוב...")
        hashes_path = os.path.join(_HERE, "trained_hashes.json")
        if os.path.exists(hashes_path):
            os.remove(hashes_path)
        train()


def chat():
    """פותח את ה-GUI — כל הלוגיקה נמצאת ב-chat.py (מקור יחיד)."""
    from chat import main
    main()


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("בודק שינויים בנתונים...")
    run_train()

    if not has_base_model():
        print("❌ לא נוצר מודל — בדוק שקיים training_data.json")
        sys.exit(1)

    print("\nפותח צ'אט...")
    chat()
