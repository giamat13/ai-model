"""
data_utils.py — עוזר משותף לסקריפטי generator שממזגים נתונים ל-training_data.json.

training_data.json הוא מקור-האמת היחיד לכל דוגמאות האימון. סקריפטי generator
(add_math_data.py, generate_naming_data.py וכו') לא כותבים קובץ נפרד משלהם —
הם ממזגים ישירות לתוכו, לפי id: id קיים מוחלף (regenerate), id חדש מתווסף.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(_HERE, "training_data.json")


def merge_into_training_data(new_articles: list[dict], source_label: str) -> None:
    """
    ממזג new_articles לתוך training_data.json, לפי id:
      - id שכבר קיים → מוחלף בתוכן החדש (regenerate של אותה קבוצה)
      - id חדש → מתווסף
    מעדכן metadata (num_items, total_words, total_characters, last_merged_at).
    """
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        main = json.load(f)

    by_id = {a["id"]: a for a in main["articles"]}
    added = replaced = 0
    for art in new_articles:
        if art["id"] in by_id:
            replaced += 1
        else:
            added += 1
        by_id[art["id"]] = art

    main["articles"] = list(by_id.values())
    main.setdefault("metadata", {})
    main["metadata"]["num_items"] = len(main["articles"])
    main["metadata"]["total_words"] = sum(len(a.get("text", "").split()) for a in main["articles"])
    main["metadata"]["total_characters"] = sum(len(a.get("text", "")) for a in main["articles"])
    main["metadata"]["last_merged_at"] = datetime.now(timezone.utc).isoformat()

    with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(main, f, ensure_ascii=False, indent=2)

    print(f"[{source_label}] מוזג ל-training_data.json: {added} חדשים, {replaced} הוחלפו "
          f"→ סה\"כ {len(main['articles']):,} מאמרים")
