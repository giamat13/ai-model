#!/usr/bin/env python3
"""Generate 100K diverse training examples for the AI model."""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any

# Hebrew dialogue intents
HEBREW_INTENTS = {
    "greeting": [
        "שלום, מה שלומך?",
        "היי, איך אתה?",
        "בוקר טוב",
        "ערב טוב",
        "לילה טוב",
    ],
    "help": [
        "אני צריך עזרה",
        "אתה יכול לעזור לי?",
        "אני נתקע",
        "איך אני עושה את זה?",
        "אפשר עזרה בבקשה?",
    ],
    "coding": [
        "איך כותבים פונקציה?",
        "מה זה variable?",
        "איך עובדת לולאה?",
        "מה זה אובייקט?",
        "איך עושים import?",
        "מה זה class?",
        "איך עובד if statement?",
        "מה זה list?",
        "איך עובד string?",
        "מה זה dictionary?",
    ],
    "math": [
        "מה זה 2 ועוד 3?",
        "כמה זה 10 פחות 4?",
        "כמה זה 5 כפול 6?",
        "כמה זה 20 חלקי 4?",
        "מה הסכום של 15 ו 25?",
        "מה ההפרש בין 50 ל 30?",
        "מה המכפלה של 7 ו 8?",
        "כמה זה 100 מחולק ב 5?",
    ],
    "knowledge": [
        "מי אתה?",
        "מה השם שלך?",
        "איפה אתה מגר?",
        "מה העבודה שלך?",
        "מה יכול אתה לעשות?",
        "מתי אתה נולדת?",
    ]
}

# English dialogue intents
ENGLISH_INTENTS = {
    "greeting": [
        "Hello, how are you?",
        "Hi there!",
        "Good morning",
        "Good evening",
        "Good night",
    ],
    "help": [
        "I need help",
        "Can you help me?",
        "I'm stuck",
        "How do I do this?",
        "Can you help please?",
    ],
    "coding": [
        "How do I write a function?",
        "What is a variable?",
        "How does a loop work?",
        "What is an object?",
        "How do I import?",
        "What is a class?",
        "How does if statement work?",
        "What is a list?",
        "How does string work?",
        "What is a dictionary?",
    ],
    "math": [
        "What is 2 plus 3?",
        "What is 10 minus 4?",
        "What is 5 times 6?",
        "What is 20 divided by 4?",
        "What is the sum of 15 and 25?",
        "What is the difference between 50 and 30?",
        "What is the product of 7 and 8?",
        "What is 100 divided by 5?",
    ],
    "knowledge": [
        "Who are you?",
        "What is your name?",
        "Where do you live?",
        "What is your job?",
        "What can you do?",
        "When were you born?",
    ]
}

# Hebrew responses
HEBREW_RESPONSES = {
    "greeting": [
        "שלום! אני כאן כדי לעזור לך.",
        "היי, בסדר! ואתה?",
        "בוקר טוב, מה חדש?",
        "ערב טוב!",
        "לילה טוב, שו״ם טוב!",
    ],
    "help": [
        "כן, אני אוכל לעזור לך.",
        "בהחלט, בואנו נפתור את זה.",
        "בטח, מה צריך?",
        "בשמחה, אני כאן בשביל זה.",
    ],
    "identity": [
        "אני עוזר AI בנוי מ-NumPy, עברית ואנגלית.",
        "שמי עוזר AI ודברי עזרה לך.",
        "אני מודל למידה קטן בשם MiniLM.",
    ],
}

# English responses
ENGLISH_RESPONSES = {
    "greeting": [
        "Hello! I'm here to help you.",
        "Hi! How can I assist you?",
        "Good morning, what can I do for you?",
        "Good evening!",
        "Good night, sleep well!",
    ],
    "help": [
        "Yes, I can help you.",
        "Absolutely, let's solve this together.",
        "Sure, what do you need?",
        "Happy to help!",
    ],
    "identity": [
        "I'm an AI assistant built from NumPy, Hebrew and English.",
        "My name is AI Helper and I help you.",
        "I'm a small learning model called MiniLM.",
    ],
}

# Python code examples
PYTHON_CODE_EXAMPLES = [
    ("def hello():\n    print('Hello')", "Function that prints hello"),
    ("x = 5\ny = 10\nprint(x + y)", "Add two variables"),
    ("for i in range(10):\n    print(i)", "Loop from 0 to 9"),
    ("numbers = [1, 2, 3, 4, 5]", "Create a list"),
    ("if x > 5:\n    print('Greater')", "If statement"),
    ("my_dict = {'name': 'John', 'age': 30}", "Create a dictionary"),
    ("def add(a, b):\n    return a + b", "Function that adds"),
    ("while True:\n    pass", "Infinite loop"),
    ("s = 'Hello World'\nprint(len(s))", "String length"),
    ("import math\nprint(math.sqrt(16))", "Import and use module"),
]

# Topics for diversification
HEBREW_TOPICS = [
    "תכנות", "עברית", "מתמטיקה", "מדע", "היסטוריה",
    "דיכדוך", "סיפור", "עזרה כללית", "טכנולוגיה", "ספרות",
    "פילוסופיה", "אנגלית", "מוזיקה", "אמנות", "ספורט",
    "בנק נתונים", "רשתות", "보안", "מערכות הפעלה", "תקשורת"
]

ENGLISH_TOPICS = [
    "Programming", "English", "Mathematics", "Science", "History",
    "Grammar", "Story", "General Help", "Technology", "Literature",
    "Philosophy", "Hebrew", "Music", "Art", "Sports",
    "Database", "Networks", "Security", "Operating Systems", "Communication"
]

def generate_hebrew_dialogue(index: int) -> Dict[str, Any]:
    """Generate a Hebrew dialogue example."""
    intent = random.choice(list(HEBREW_INTENTS.keys()))
    question = random.choice(HEBREW_INTENTS[intent])

    if intent == "greeting":
        answer = random.choice(HEBREW_RESPONSES["greeting"])
    elif intent == "help":
        answer = random.choice(HEBREW_RESPONSES["help"])
    elif intent == "knowledge":
        answer = random.choice(HEBREW_RESPONSES["identity"])
    else:
        answer = f"זה שאלה טובה על {random.choice(HEBREW_TOPICS)}."

    topic = random.choice(HEBREW_TOPICS)

    return {
        "id": f"he_dialogue_{index:06d}",
        "source": "generated_dialogue",
        "language": "he",
        "topic": topic,
        "intent": intent,
        "text": f"User: {question} Model: {answer}"
    }

def generate_english_dialogue(index: int) -> Dict[str, Any]:
    """Generate an English dialogue example."""
    intent = random.choice(list(ENGLISH_INTENTS.keys()))
    question = random.choice(ENGLISH_INTENTS[intent])

    if intent == "greeting":
        answer = random.choice(ENGLISH_RESPONSES["greeting"])
    elif intent == "help":
        answer = random.choice(ENGLISH_RESPONSES["help"])
    elif intent == "knowledge":
        answer = random.choice(ENGLISH_RESPONSES["identity"])
    else:
        answer = f"That's a good question about {random.choice(ENGLISH_TOPICS)}."

    topic = random.choice(ENGLISH_TOPICS)

    return {
        "id": f"en_dialogue_{index:06d}",
        "source": "generated_dialogue",
        "language": "en",
        "topic": topic,
        "intent": intent,
        "text": f"User: {question} Model: {answer}"
    }

def generate_math_example(index: int) -> Dict[str, Any]:
    """Generate a math training example."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    operator = random.choice(['+', '-', '*', '/', '//', '%'])

    # Hebrew or English
    is_hebrew = random.random() < 0.5

    if operator == '+':
        op_words = ["ועוד", "פלוס", "הסכום של"] if is_hebrew else ["plus", "add"]
        op_text = random.choice(op_words)
        if is_hebrew:
            text = f"User: מה {op_text} {a} ו {b}? Model: <calc> +"
        else:
            text = f"User: What is {a} {op_text} {b}? Model: <calc> +"
    elif operator == '-':
        op_words = ["פחות", "מינוס", "ההפרש"] if is_hebrew else ["minus", "subtract"]
        op_text = random.choice(op_words)
        if is_hebrew:
            text = f"User: מה {a} {op_text} {b}? Model: <calc> -"
        else:
            text = f"User: What is {a} {op_text} {b}? Model: <calc> -"
    elif operator == '*':
        op_words = ["כפול", "פעמים", "פי", "המכפלה"] if is_hebrew else ["times", "multiply"]
        op_text = random.choice(op_words)
        if is_hebrew:
            text = f"User: {a} {op_text} {b}? Model: <calc> *"
        else:
            text = f"User: {a} {op_text} {b}? Model: <calc> *"
    else:  # division
        b = random.randint(1, 20)  # Avoid very large numbers
        op_words = ["חלקי", "מחולק"] if is_hebrew else ["divided by"]
        op_text = random.choice(op_words)
        if is_hebrew:
            text = f"User: {a} {op_text} {b}? Model: <calc> /"
        else:
            text = f"User: {a} {op_text} {b}? Model: <calc> /"

    return {
        "id": f"math_{index:06d}",
        "source": "generated_math",
        "language": "he" if is_hebrew else "en",
        "topic": "arithmetic",
        "intent": "math",
        "text": text
    }

def generate_code_example(index: int) -> Dict[str, Any]:
    """Generate a code training example."""
    code, description = random.choice(PYTHON_CODE_EXAMPLES)
    is_hebrew = random.random() < 0.5

    if is_hebrew:
        text = f"User: כתוב קוד שיעשה {description} Model: ```python\n{code}\n```"
    else:
        text = f"User: Write code to {description} Model: ```python\n{code}\n```"

    return {
        "id": f"code_{index:06d}",
        "source": "generated_code",
        "language": "he" if is_hebrew else "en",
        "topic": "programming",
        "intent": "coding",
        "text": text
    }

def generate_knowledge_example(index: int) -> Dict[str, Any]:
    """Generate a knowledge/fact example."""
    facts_he = [
        ("ראשית: עברית היא שפה סמיטית עתיקה.", "Hebrew is an ancient Semitic language."),
        ("Python היא שפת תכנות פופולרית.", "Python is a popular programming language."),
        ("מתמטיקה היא מדע של מספרים וצורות.", "Mathematics is the science of numbers and shapes."),
        ("מדע המחשב עוסק בעיבוד נתונים.", "Computer science deals with data processing."),
        ("AI הוא תחום מהיר בתכנות מודרני.", "AI is a rapidly growing field in modern programming."),
    ]

    if random.random() < 0.5:
        fact_he, fact_en = random.choice(facts_he)
        text = f"User: ספר לי עובדה. Model: {fact_he}"
    else:
        fact_he, fact_en = random.choice(facts_he)
        text = f"User: Tell me a fact. Model: {fact_en}"

    return {
        "id": f"knowledge_{index:06d}",
        "source": "generated_knowledge",
        "language": "he" if "זה" in text else "en",
        "topic": "general_knowledge",
        "intent": "knowledge",
        "text": text
    }

def generate_training_data(num_examples: int = 100000) -> List[Dict[str, Any]]:
    """Generate diverse training examples."""
    examples = []

    print(f"Generating {num_examples:,} training examples...")

    # Distribution:
    # 30% Hebrew dialogue
    # 25% English dialogue
    # 15% Math examples
    # 15% Code examples
    # 15% Knowledge examples

    counts = {
        'hebrew_dialogue': int(num_examples * 0.30),
        'english_dialogue': int(num_examples * 0.25),
        'math': int(num_examples * 0.15),
        'code': int(num_examples * 0.15),
        'knowledge': int(num_examples * 0.15),
    }

    index = 0

    # Hebrew dialogue
    print(f"  Generating {counts['hebrew_dialogue']:,} Hebrew dialogue examples...")
    for i in range(counts['hebrew_dialogue']):
        examples.append(generate_hebrew_dialogue(index))
        index += 1
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1:,} Hebrew dialogues generated")

    # English dialogue
    print(f"  Generating {counts['english_dialogue']:,} English dialogue examples...")
    for i in range(counts['english_dialogue']):
        examples.append(generate_english_dialogue(index))
        index += 1
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1:,} English dialogues generated")

    # Math examples
    print(f"  Generating {counts['math']:,} math examples...")
    for i in range(counts['math']):
        examples.append(generate_math_example(index))
        index += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} math examples generated")

    # Code examples
    print(f"  Generating {counts['code']:,} code examples...")
    for i in range(counts['code']):
        examples.append(generate_code_example(index))
        index += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} code examples generated")

    # Knowledge examples
    print(f"  Generating {counts['knowledge']:,} knowledge examples...")
    for i in range(counts['knowledge']):
        examples.append(generate_knowledge_example(index))
        index += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} knowledge examples generated")

    return examples

def save_training_data(examples: List[Dict[str, Any]], output_file: str = "large_training_data.json"):
    """Save generated training data to file."""
    print(f"\nPreparing to save {len(examples):,} examples...")

    # Create metadata
    total_words = sum(len(item["text"].split()) for item in examples)
    total_chars = sum(len(item["text"]) for item in examples)

    data = {
        "metadata": {
            "name": "100K Generated Training Dataset",
            "version": "1.0",
            "description": "Large-scale generated training data with Hebrew/English dialogue, math, code, and knowledge examples",
            "language": "he+en",
            "generated_at": datetime.now().isoformat(),
            "num_items": len(examples),
            "total_words": total_words,
            "total_characters": total_chars,
            "distribution": {
                "hebrew_dialogue": 0.30,
                "english_dialogue": 0.25,
                "math": 0.15,
                "code": 0.15,
                "knowledge": 0.15,
            }
        },
        "articles": examples
    }

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(examples):,} examples to {output_file}")
    print(f"  Total words: {total_words:,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")

def merge_with_existing(output_file: str = "training_data.json", new_file: str = "large_training_data.json"):
    """Merge new training data with existing data."""
    # Load existing training data
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_articles = existing_data.get("articles", [])
        print(f"\nLoaded {len(existing_articles):,} existing articles from {output_file}")
    else:
        existing_articles = []
        print(f"\nNo existing {output_file} found, starting fresh")

    # Load new training data
    with open(new_file, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    new_articles = new_data.get("articles", [])

    # Merge (avoid duplicates by ID)
    by_id = {item.get("id"): item for item in existing_articles if item.get("id")}
    for item in new_articles:
        by_id[item["id"]] = item

    merged_articles = list(by_id.values())

    # Calculate statistics
    total_words = sum(len(item["text"].split()) for item in merged_articles)
    total_chars = sum(len(item["text"]) for item in merged_articles)

    # Save merged data
    merged_data = {
        "metadata": {
            "name": "Merged Training Dataset (100K+ examples)",
            "version": "1.0",
            "description": "Combined local and generated training data",
            "language": "he+en",
            "generated_at": datetime.now().isoformat(),
            "num_items": len(merged_articles),
            "total_words": total_words,
            "total_characters": total_chars,
        },
        "articles": merged_articles
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Merged data saved to {output_file}")
    print(f"  Total articles: {len(merged_articles):,}")
    print(f"  Total words: {total_words:,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")

    return output_file

def main():
    import sys

    # Check if merge argument is provided
    merge = "--merge" in sys.argv

    # Generate 100K examples
    examples = generate_training_data(100000)

    # Save to temporary file
    save_training_data(examples, "large_training_data.json")

    # Merge with existing if requested
    if merge:
        merge_with_existing()
        print("\n✓ All training data merged and ready for training!")
    else:
        print("\n✓ Generated 100K examples saved to large_training_data.json")
        print("  To merge with existing training_data.json, run:")
        print("  python generate_100k_training_data.py --merge")

if __name__ == "__main__":
    main()
