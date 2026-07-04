"""check_training.py — בדיקה מהירה של מצב האימון: כמה מאמרים נכנסו ל-hashes
לעומת כמה יש כרגע ב-training_data.json (מקור-האמת היחיד)."""
import json
import os

if os.path.exists('trained_hashes.json'):
    with open('trained_hashes.json', 'r', encoding='utf-8') as f:
        hashes = json.load(f)
    print(f'Total trained articles: {len(hashes)}')

    naming_count = sum(1 for k in hashes.keys() if k.startswith('naming_'))
    print(f'Naming articles in hashes: {naming_count}')
else:
    print('No trained_hashes.json found')

with open('training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    articles = data['articles']
    print(f'Total articles in training_data.json: {len(articles)}')

    naming = sum(1 for a in articles if a.get('id', '').startswith('naming_'))
    calc = sum(1 for a in articles if a.get('id', '').startswith('calc_'))
    print(f'  naming articles: {naming}')
    print(f'  math (calc_) articles: {calc}')
