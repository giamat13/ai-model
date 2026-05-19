import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "better_code_examples.json"), "r", encoding="utf-8") as f:
    new_data = json.load(f)

with open(os.path.join(_HERE, "training_data.json"), "r", encoding="utf-8") as f:
    main_data = json.load(f)

main_data["articles"].extend(new_data["better_code_examples"])
main_data["metadata"]["num_items"] = len(main_data["articles"])

with open(os.path.join(_HERE, "training_data.json"), "w", encoding="utf-8") as f:
    json.dump(main_data, f, ensure_ascii=False, indent=2)

print(f"Added {len(new_data['better_code_examples'])} better examples")
print(f"Total: {main_data['metadata']['num_items']}")
