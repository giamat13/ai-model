#!/usr/bin/env python3
"""Generate richer, longer training examples with realistic context."""

import json
import random
from datetime import datetime

# Real knowledge base (Hebrew and English)
KNOWLEDGE_BASE = [
    {
        "he": "מדעי המחשב הוא תחום המתעסק בתיאוריה של חישוב, אוטומציה ומידע. הוא כולל תת-תחומים כמו אלגוריתמים, מבני נתונים, בסיסי נתונים, רשתות מחשבים וביטחון מידע.",
        "en": "Computer Science is the study of computation, automation, and information. It includes subfields such as algorithms, data structures, databases, computer networks, and information security."
    },
    {
        "he": "Python היא שפת תכנות דינמית המפורסמת בקלות הלמידה שלה. היא משמשת לדברים רבים: מדע נתונים, פיתוח אתרים, אוטומציה, בינה מלאכותית ועוד.",
        "en": "Python is a dynamic programming language famous for its ease of learning. It is used for many things: data science, web development, automation, artificial intelligence, and more."
    },
    {
        "he": "למידת מכונה (Machine Learning) היא תחום בבינה מלאכותית שבו מערכות מחשב לומדות מנתונים ללא התכנתה מראש של כללים מפורשים.",
        "en": "Machine Learning is a field of artificial intelligence where computer systems learn from data without being explicitly programmed with rules."
    },
    {
        "he": "בסיס נתונים (Database) הוא אוסף מנוהל של נתונים המאוחסנים ונגישים בדרך אלקטרונית. דוגמאות: MySQL, PostgreSQL, MongoDB.",
        "en": "A Database is a managed collection of data stored and accessible electronically. Examples: MySQL, PostgreSQL, MongoDB."
    },
    {
        "he": "אלגוריתם הוא סדרה של צעדים/הוראות להפתרת בעיה מסוימת. אלגוריתמים טובים הם מהירים, יעילים ופחות משתמשים בזיכרון.",
        "en": "An algorithm is a sequence of steps/instructions to solve a specific problem. Good algorithms are fast, efficient, and use less memory."
    },
    {
        "he": "קריפטוגרפיה היא מדע הצפנה. היא משמשת להגן על מידע סודי בתקשורת דיגיטלית ובאחסון נתונים.",
        "en": "Cryptography is the science of encryption. It is used to protect secret information in digital communication and data storage."
    },
]

# Multi-turn conversations (more realistic)
CONVERSATIONS = [
    {
        "he": [
            ("איך אני מתחיל עם Python?", "אתה יכול להתחיל בהורדת Python מהאתר הרשמי, ואז ללמוד את הבסיס: משתנים, לולאות ותנאים. יש הרבה ספרים וקורסים חינמיים בחינם."),
            ("מה יותר קל: Python או Java?", "Python נחשבת לקלה יותר ללומדים חדשים בגלל התחביר הפשוט שלה. Java קצת יותר מורכבת אבל חזקה יותר לאפליקציות גדולות."),
        ],
        "en": [
            ("How do I start with Python?", "You can start by downloading Python from the official website, then learn the basics: variables, loops, and conditionals. There are many free books and courses available."),
            ("Which is easier: Python or Java?", "Python is considered easier for beginners because of its simple syntax. Java is somewhat more complex but more powerful for large applications."),
        ]
    },
    {
        "he": [
            ("מה זה מודל AI?", "מודל AI הוא אלגוריתם שנלמד מנתונים והוא יכול לעשות תחזוקות או סיווגים. למשל, מודל שמזהה חתולים בתמונות."),
            ("איך מודלים למדים?", "מודלים למדים דרך תהליך הנקרא 'אימון' שבו הם מקבלים הרבה דוגמאות וכל פעם הם משפרים את הביצוע שלהם."),
        ],
        "en": [
            ("What is an AI model?", "An AI model is an algorithm learned from data that can make predictions or classifications. For example, a model that identifies cats in images."),
            ("How do models learn?", "Models learn through a process called 'training' where they receive many examples and gradually improve their performance."),
        ]
    },
]

# Practical examples with explanations
PRACTICAL_EXAMPLES = [
    {
        "he": "קוד לקריאת קובץ בPython:\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(content)\n```\nזה פותח את הקובץ, קורא את התוכן, מדפיס אותו, וסוגר את הקובץ באופן אוטומטי.",
        "en": "Code to read a file in Python:\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(content)\n```\nThis opens the file, reads the content, prints it, and automatically closes the file."
    },
    {
        "he": "לולאה לעיבוד רשימה:\n```python\nnumbers = [1, 2, 3, 4, 5]\nfor num in numbers:\n    print(num * 2)\n```\nזה מדפיס כל מספר כפול 2. פלט: 2, 4, 6, 8, 10",
        "en": "Loop to process a list:\n```python\nnumbers = [1, 2, 3, 4, 5]\nfor num in numbers:\n    print(num * 2)\n```\nThis prints each number multiplied by 2. Output: 2, 4, 6, 8, 10"
    },
    {
        "he": "פונקציה עם פרמטרים:\n```python\ndef greet(name, age):\n    return f'שלום {name}, אתה בן {age}'\n\nprint(greet('דוד', 30))\n```\nזה מדפיס: 'שלום דוד, אתה בן 30'",
        "en": "Function with parameters:\n```python\ndef greet(name, age):\n    return f'Hello {name}, you are {age} years old'\n\nprint(greet('David', 30))\n```\nThis prints: 'Hello David, you are 30 years old'"
    },
]

# Complex scenarios (reasoning)
SCENARIOS = [
    {
        "he": "שאלה: אני רוצה לבנות אתר. מה צריך לדעת?\nתשובה: אתה צריך לדעת:\n1. HTML/CSS לעיצוב הדף\n2. JavaScript לאינטראקציה\n3. Backend language כמו Python/Node.js לשרת\n4. בסיס נתונים לאחסון מידע\n5. ביטחון (HTTPS, validation, protection מ-hacking)\nזה תהליך שלם שדורש כמה שבועות ללמוד.",
        "en": "Question: I want to build a website. What do I need to know?\nAnswer: You need to know:\n1. HTML/CSS for page design\n2. JavaScript for interaction\n3. Backend language like Python/Node.js for server\n4. Database for storing information\n5. Security (HTTPS, validation, protection from hacking)\nIt's a complete process that requires several weeks to learn."
    },
    {
        "he": "שאלה: איך בוחרים שפת תכנות לפרויקט?\nתשובה: זה תלוי בדרישות:\n- מהירות? → C++, Rust\n- קלות פיתוח? → Python, JavaScript\n- מובייל? → Swift (iOS), Kotlin (Android)\n- Web? → JavaScript, Python, PHP\n- ביטחון? → Rust, Ada\nתמיד בחרו בשפה שהיא פופולרית ויש לה קהילה טובה.",
        "en": "Question: How do you choose a programming language for a project?\nAnswer: It depends on requirements:\n- Speed? → C++, Rust\n- Easy development? → Python, JavaScript\n- Mobile? → Swift (iOS), Kotlin (Android)\n- Web? → JavaScript, Python, PHP\n- Security? → Rust, Ada\nAlways choose a language that is popular and has a good community."
    },
]

def generate_rich_training_data(num_examples: int = 50000) -> list:
    """Generate richer training data with longer, more realistic examples."""
    examples = []
    example_id = 0

    print(f"Generating {num_examples:,} richer training examples...")

    # Knowledge-based examples
    print("  Generating knowledge examples...")
    for i in range(num_examples // 5):
        kb = random.choice(KNOWLEDGE_BASE)
        is_hebrew = random.random() < 0.5
        text = f"User: ספר לי על {random.choice(['מדעי המחשב', 'תכנות', 'נתונים'])}\nModel: {kb['he']}" if is_hebrew else f"User: Tell me about {random.choice(['computer science', 'programming', 'data'])}\nModel: {kb['en']}"

        examples.append({
            "id": f"knowledge_rich_{example_id:06d}",
            "source": "generated_knowledge_rich",
            "language": "he" if is_hebrew else "en",
            "type": "knowledge",
            "text": text
        })
        example_id += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} knowledge examples")

    # Multi-turn conversations
    print("  Generating conversation examples...")
    for i in range(num_examples // 5):
        conv = random.choice(CONVERSATIONS)
        is_hebrew = random.random() < 0.5
        conv_pair = random.choice(conv["he" if is_hebrew else "en"])
        q, a = conv_pair
        text = f"User: {q}\nModel: {a}"

        examples.append({
            "id": f"conversation_rich_{example_id:06d}",
            "source": "generated_conversation_rich",
            "language": "he" if is_hebrew else "en",
            "type": "conversation",
            "text": text
        })
        example_id += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} conversation examples")

    # Practical examples
    print("  Generating practical examples...")
    for i in range(num_examples // 5):
        example = random.choice(PRACTICAL_EXAMPLES)
        is_hebrew = random.random() < 0.5
        content = example["he"] if is_hebrew else example["en"]
        text = f"User: בואנו נלמד קוד\nModel: {content}" if is_hebrew else f"User: Let's learn code\nModel: {content}"

        examples.append({
            "id": f"practical_rich_{example_id:06d}",
            "source": "generated_practical_rich",
            "language": "he" if is_hebrew else "en",
            "type": "practical",
            "text": text
        })
        example_id += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} practical examples")

    # Scenario-based reasoning
    print("  Generating scenario examples...")
    for i in range(num_examples // 5):
        scenario = random.choice(SCENARIOS)
        is_hebrew = random.random() < 0.5
        content = scenario["he"] if is_hebrew else scenario["en"]
        text = f"Model: {content}"

        examples.append({
            "id": f"scenario_rich_{example_id:06d}",
            "source": "generated_scenario_rich",
            "language": "he" if is_hebrew else "en",
            "type": "scenario",
            "text": text
        })
        example_id += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} scenario examples")

    # Remaining quota - dialogue with longer context
    print(f"  Generating {num_examples - example_id:,} dialogue examples...")
    remaining = num_examples - example_id
    for i in range(remaining):
        is_hebrew = random.random() < 0.5
        if is_hebrew:
            questions = ["מה זה?", "איך זה עובד?", "למה זה חשוב?", "איך משתמשים בזה?"]
            topics = ["תכנות", "מתמטיקה", "מדע", "היסטוריה", "טכנולוגיה"]
            q = random.choice(questions) + " " + random.choice(topics)
            answers = ["זו שאלה טובה מאוד.", "בדיוק המשהו חשוב לדעת.", "אני שמח ששאלת.", "בואנו נחקור את זה יחד."]
            a = random.choice(answers)
        else:
            questions = ["What is?", "How does it work?", "Why is it important?", "How do we use it?"]
            topics = ["programming", "mathematics", "science", "history", "technology"]
            q = random.choice(questions) + " " + random.choice(topics)
            answers = ["That's a very good question.", "That's something important to know.", "I'm glad you asked.", "Let's explore that together."]
            a = random.choice(answers)

        text = f"User: {q}\nModel: {a}"
        examples.append({
            "id": f"dialogue_rich_{example_id:06d}",
            "source": "generated_dialogue_rich",
            "language": "he" if is_hebrew else "en",
            "type": "dialogue",
            "text": text
        })
        example_id += 1
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1:,} dialogue examples")

    return examples

def save_and_merge(examples: list, output_file: str = "training_data.json"):
    """Save and merge with existing data."""
    print(f"\nMerging {len(examples):,} richer examples with existing data...")

    # Load existing
    with open(output_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    existing_articles = existing_data.get("articles", [])

    # Merge by ID
    by_id = {item.get("id"): item for item in existing_articles if item.get("id")}
    for item in examples:
        by_id[item["id"]] = item

    merged_articles = list(by_id.values())

    # Stats
    total_words = sum(len(item["text"].split()) for item in merged_articles)
    total_chars = sum(len(item["text"]) for item in merged_articles)

    # Save
    data = {
        "metadata": {
            "name": "Enhanced Training Dataset with Rich Examples",
            "version": "2.0",
            "description": "Training data with realistic conversations, knowledge base, code examples, and reasoning scenarios",
            "language": "he+en",
            "generated_at": datetime.now().isoformat(),
            "num_items": len(merged_articles),
            "total_words": total_words,
            "total_characters": total_chars,
        },
        "articles": merged_articles
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Merged {len(merged_articles):,} examples to {output_file}")
    print(f"  Total words: {total_words:,}")
    print(f"  File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    import os
    examples = generate_rich_training_data(50000)
    save_and_merge(examples)
