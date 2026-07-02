#!/usr/bin/env python3
"""Generate 1000 new training examples for the AI model in Hebrew and English."""

import json
import random
from datetime import datetime

# Hebrew dialogue templates - diverse conversation scenarios
HEBREW_DIALOGUE_TEMPLATES = [
    ("אני צריך עזרה בקידוד", "בשמחה! באיזה שפת תכנות אתה עובד?"),
    ("איך יוצרים משתנה ב-Python?", "משתנה ב-Python נוצר בעזרת השמה: variable_name = value"),
    ("מה זה לולאה?", "לולאה היא מבנה בתכנות שמחזור פקודות מסוימות כל עוד תנאי מסוים מתקיים."),
    ("איך אני יוצר פונקציה?", "פונקציה נוצרת עם def function_name(): ואז הגוף של הפונקציה."),
    ("מה הביעו קבוע?", "ביטוי קבוע היא דרך להגדיר תבנית לחיפוש וטיפול בטקסט."),
    ("איך קוראים לדאטא בייס?", "קראנו לדאטא בייס עם שאילתות SQL כמו SELECT, INSERT, UPDATE, DELETE."),
    ("מה זה API?", "API הוא ממשק תכנות שמאפשר לתוכניות להתקשר ביניהן."),
    ("איך אני מוסיף אלמנט לרשימה?", "ברשימה בPython אתה משתמש ב-list.append(element) להוספת אלמנט."),
    ("מה הוא דיקשונרי?", "דיקשונרי הוא מבנה נתונים שמאחסן זוגות של מפתח וערך."),
    ("איך אני כותב תנאי?", "תנאי נכתב עם if, elif, ו-else, כמו: if x > 5: print('גדול')"),
]

ENGLISH_DIALOGUE_TEMPLATES = [
    ("I need help with coding", "Sure! What programming language are you working with?"),
    ("How do I create a variable in Python?", "A variable in Python is created by assignment: variable_name = value"),
    ("What is a loop?", "A loop is a control structure that repeats code while a condition is true."),
    ("How do I create a function?", "A function is created with def function_name(): followed by the function body."),
    ("What is a regex?", "A regex is a pattern used for searching and manipulating text strings."),
    ("How do I query a database?", "You query a database using SQL statements like SELECT, INSERT, UPDATE, DELETE."),
    ("What is an API?", "An API is an interface that allows different software programs to communicate."),
    ("How do I add to a list?", "In Python, use list.append(element) to add an element to a list."),
    ("What is a dictionary?", "A dictionary is a data structure that stores key-value pairs."),
    ("How do I write a conditional?", "Use if, elif, and else statements like: if x > 5: print('greater')"),
]

# Topics for diversification
HEBREW_TOPICS = [
    "תכנות",
    "עברית",
    "מתמטיקה",
    "מדע",
    "היסטוריה",
    "דיכדוך",
    "סיפור",
    "עזרה כללית",
    "טכנולוגיה",
    "ספרות",
    "פילוסופיה",
    "אנגלית",
    "מוזיקה",
    "אמנות",
    "ספורט"
]

ENGLISH_TOPICS = [
    "Programming",
    "English",
    "Mathematics",
    "Science",
    "History",
    "Grammar",
    "Story",
    "General Help",
    "Technology",
    "Literature",
    "Philosophy",
    "Hebrew",
    "Music",
    "Art",
    "Sports"
]

# Response templates for generating diverse answers
HEBREW_RESPONSES = [
    "כמובן, אשמח לעזור.",
    "זו שאלה טובה.",
    "בהחלט, אני יכול להסביר.",
    "בדיוק כמו שרצית לשמוע.",
    "שאלה מעניינת מאוד.",
    "אני מסכים איתך לחלוטין.",
    "הנקודה שלך נכונה מאוד.",
    "אפשר לחשוב על זה בדרכים רבות.",
    "זה תלוי בהקשר.",
    "חשוב להבין את ההנחות הבסיסיות.",
]

ENGLISH_RESPONSES = [
    "Of course, I'd be happy to help.",
    "That's a good question.",
    "Absolutely, I can explain.",
    "Exactly as you wanted to hear.",
    "That's a very interesting question.",
    "I completely agree with you.",
    "Your point is very valid.",
    "We can think about this in many ways.",
    "It depends on the context.",
    "It's important to understand the basics.",
]

def generate_hebrew_dialogue(index):
    """Generate a Hebrew dialogue example."""
    question, answer = random.choice(HEBREW_DIALOGUE_TEMPLATES)
    topic = random.choice(HEBREW_TOPICS)

    return {
        "id": f"he_gen_{index:04d}",
        "source": "dialogue",
        "topic": topic,
        "text": f"User: {question} Model: {answer}"
    }

def generate_english_dialogue(index):
    """Generate an English dialogue example."""
    question, answer = random.choice(ENGLISH_DIALOGUE_TEMPLATES)
    topic = random.choice(ENGLISH_TOPICS)

    return {
        "id": f"en_gen_{index:04d}",
        "source": "dialogue",
        "topic": topic,
        "text": f"User: {question} Model: {answer}"
    }

def generate_mixed_dialogue(index):
    """Generate a mixed Hebrew-English dialogue."""
    if random.random() < 0.5:
        # Hebrew question with English answer
        question = random.choice([q for q, _ in HEBREW_DIALOGUE_TEMPLATES])
        answer = random.choice([a for _, a in ENGLISH_DIALOGUE_TEMPLATES])
        topic = random.choice(HEBREW_TOPICS)
    else:
        # English question with Hebrew answer
        question = random.choice([q for q, _ in ENGLISH_DIALOGUE_TEMPLATES])
        answer = random.choice([a for _, a in HEBREW_DIALOGUE_TEMPLATES])
        topic = random.choice(ENGLISH_TOPICS)

    return {
        "id": f"mixed_gen_{index:04d}",
        "source": "dialogue",
        "topic": topic,
        "text": f"User: {question} Model: {answer}"
    }

def generate_training_data(num_examples=1000):
    """Generate new training examples."""
    examples = []

    for i in range(num_examples):
        # Roughly 40% Hebrew, 40% English, 20% mixed
        rand = random.random()
        if rand < 0.4:
            examples.append(generate_hebrew_dialogue(i))
        elif rand < 0.8:
            examples.append(generate_english_dialogue(i))
        else:
            examples.append(generate_mixed_dialogue(i))

    return examples

def main():
    # Load existing data
    with open("training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_articles = data.get("articles", [])
    print(f"Existing articles: {len(existing_articles)}")

    # Generate new examples
    print("Generating 1000 new training examples...")
    new_examples = generate_training_data(1000)

    # Combine
    all_articles = existing_articles + new_examples

    # Calculate statistics
    total_words = sum(len(item["text"].split()) for item in all_articles)
    total_chars = sum(len(item["text"]) for item in all_articles)

    # Update metadata
    data["metadata"]["num_items"] = len(all_articles)
    data["metadata"]["total_words"] = total_words
    data["metadata"]["total_characters"] = total_chars
    data["articles"] = all_articles

    # Save
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nNew total items: {len(all_articles)}")
    print(f"Total words: {total_words}")
    print(f"Total characters: {total_chars}")
    print("Training data saved to training_data.json")

if __name__ == "__main__":
    main()
