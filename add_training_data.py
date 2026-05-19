"""
add_training_data.py - מוסיף נתוני אימון חדשים לקובץ הראשי
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# קריאת הנתונים החדשים
with open(os.path.join(_HERE, "new_training_examples.json"), "r", encoding="utf-8") as f:
    new_data = json.load(f)

# קריאת הקובץ הראשי
with open(os.path.join(_HERE, "training_data.json"), "r", encoding="utf-8") as f:
    main_data = json.load(f)

# הוספת הנתונים החדשים
main_data["articles"].extend(new_data["new_training_data"])

# עדכון המטא-דאטה
main_data["metadata"]["num_items"] = len(main_data["articles"])
main_data["metadata"]["version"] = "2.1"
main_data["metadata"]["description"] += " + קוד פייתון והסברים טכניים"

# שמירה
with open(os.path.join(_HERE, "training_data.json"), "w", encoding="utf-8") as f:
    json.dump(main_data, f, ensure_ascii=False, indent=2)

print(f"Added {len(new_data['new_training_data'])} new examples")
print(f"Total examples: {main_data['metadata']['num_items']}")
