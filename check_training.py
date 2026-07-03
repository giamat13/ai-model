import json
import os

# Check what's in trained_hashes.json
if os.path.exists('trained_hashes.json'):
    with open('trained_hashes.json', 'r', encoding='utf-8') as f:
        hashes = json.load(f)
    print(f'Total trained articles: {len(hashes)}')

    # Check how many are naming articles
    naming_count = sum(1 for k in hashes.keys() if k.startswith('naming_'))
    print(f'Naming articles in hashes: {naming_count}')
else:
    print('No trained_hashes.json found')

# Check naming data file
with open('naming_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'Total naming articles available: {len(data["articles"])}')

# Check new_training_examples
with open('new_training_examples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    total = len(data.get("new_training_data", []))
    naming = sum(1 for item in data.get("new_training_data", []) if "naming" in item.get("topic", "").lower())
    print(f'New examples total: {total}, naming: {naming}')
