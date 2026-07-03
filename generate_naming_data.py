#!/usr/bin/env python3
"""Generate creative naming examples for various business types and entities."""

import json
import random
from datetime import datetime

# Business types and entity categories
BUSINESS_TYPES_HE = [
    "חנות לחם", "קפה", "מסעדה", "חנות פרחים", "ספרייה",
    "ספר עם פרחים", "מכבסה", "חנות בגדים", "חנות רהיטים",
    "חנות צעצועים", "מלון", "בריכה", "חדר כושר", "סטודיו ריקוד",
    "בית קולנוע", "גלריית אמנות", "חנות תכשיטים", "מוזיאון",
    "בית קפה", "אולם אירועים", "מושכה מיוחדת"
]

BUSINESS_TYPES_EN = [
    "coffee shop", "bakery", "restaurant", "flower shop", "bookstore",
    "laundry", "clothing store", "furniture store", "toy store",
    "hotel", "swimming pool", "gym", "dance studio",
    "cinema", "art gallery", "jewelry store", "museum",
    "bar", "event hall", "photography studio"
]

# Name templates - Hebrew
NAME_TEMPLATES_HE = [
    "{x} הזהב",
    "פינת ה{x}",
    "{x} של פעם",
    "בית {x}",
    "{x} ומעלה",
    "מלון {x}",
    "גן ה{x}",
    "הסוד של {x}",
    "טרייד {x}",
    "עולם ה{x}",
    "צו ה{x}",
    "הטוב ביותר {x}",
    "עיר {x}",
    "כנפי {x}",
    "אל {x}",
]

# Name templates - English
NAME_TEMPLATES_EN = [
    "The Golden {x}",
    "{x} Corner",
    "{x} House",
    "Artisan {x}",
    "{x} Haven",
    "The Cozy {x}",
    "Classic {x}",
    "Prime {x}",
    "Elite {x}",
    "Urban {x}",
    "Rustic {x}",
    "The Wise {x}",
    "Legacy {x}",
    "{x} Paradise",
    "Gateway to {x}",
]

# Additional name ideas - Hebrew
ADDITIONAL_NAMES_HE = [
    "טעם של שמחה", "מקום הנוחות", "דברים טובים", "חלום חי",
    "הים של טעם", "הטבע בעיר", "כוח הרוח", "סוד השמחה",
    "בית הטוב", "חלל של אור", "גן הפרחים", "קול הלב",
    "אור הדרך", "ריח של בוקר", "מקום מיוחד", "חיים חדשים",
]

# Additional name ideas - English
ADDITIONAL_NAMES_EN = [
    "Flavor of Joy", "Place of Comfort", "Good Things", "Living Dream",
    "Sea of Taste", "Nature in the City", "Power of Spirit", "Secret of Joy",
    "House of Good", "Space of Light", "Garden of Flowers", "Voice of Heart",
    "Light of the Way", "Scent of Morning", "Special Place", "New Life",
]

# Adjectives for creative variations - Hebrew
ADJECTIVES_HE = [
    "טוב", "יפה", "חכם", "בטוח", "חם", "חדש", "פרוח",
    "זהוב", "כחול", "אדום", "לבן", "שמח", "שקט",
]

# Adjectives for creative variations - English
ADJECTIVES_EN = [
    "good", "beautiful", "smart", "safe", "warm", "new", "blooming",
    "golden", "blue", "red", "white", "happy", "peaceful",
]


def generate_naming_examples_he(num_examples: int) -> list:
    """Generate Hebrew naming examples."""
    examples = []
    example_id = 0

    print(f"  Generating {num_examples:,} Hebrew naming examples...")

    for i in range(num_examples):
        business = random.choice(BUSINESS_TYPES_HE)

        # Generate 5 name suggestions for each business
        names = []

        # Use templates
        num_from_template = min(3, random.randint(2, 4))
        for _ in range(num_from_template):
            template = random.choice(NAME_TEMPLATES_HE)
            names.append(template.format(x=business))

        # Add additional creative names
        for _ in range(max(2, 5 - len(names))):
            if random.random() < 0.5:
                names.append(random.choice(ADDITIONAL_NAMES_HE))
            else:
                adj = random.choice(ADJECTIVES_HE)
                names.append(f"{business} {adj}")

        # Remove duplicates
        names = list(dict.fromkeys(names))[:5]

        names_str = " . ".join(names)
        text = f"User: תן שמות יצירתיים ל{business} Model: הנה רעיונות : {names_str} ."

        examples.append({
            "id": f"naming_he_{example_id:06d}",
            "source": "generated_naming_he",
            "language": "he",
            "type": "naming",
            "text": text
        })
        example_id += 1

        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} Hebrew naming examples")

    return examples, example_id


def generate_naming_examples_en(num_examples: int, start_id: int = 0) -> list:
    """Generate English naming examples."""
    examples = []
    example_id = start_id

    print(f"  Generating {num_examples:,} English naming examples...")

    for i in range(num_examples):
        business = random.choice(BUSINESS_TYPES_EN)

        # Generate 5 name suggestions for each business
        names = []

        # Use templates
        num_from_template = min(3, random.randint(2, 4))
        for _ in range(num_from_template):
            template = random.choice(NAME_TEMPLATES_EN)
            names.append(template.format(x=business))

        # Add additional creative names
        for _ in range(max(2, 5 - len(names))):
            if random.random() < 0.5:
                names.append(random.choice(ADDITIONAL_NAMES_EN))
            else:
                adj = random.choice(ADJECTIVES_EN)
                names.append(f"{adj.capitalize()} {business.capitalize()}")

        # Remove duplicates and clean up
        names = list(dict.fromkeys(names))[:5]

        names_str = " . ".join(names)
        text = f"User: give creative names for a {business} Model: Here are some ideas : {names_str} ."

        examples.append({
            "id": f"naming_en_{example_id:06d}",
            "source": "generated_naming_en",
            "language": "en",
            "type": "naming",
            "text": text
        })
        example_id += 1

        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,} English naming examples")

    return examples


def generate_naming_data(num_examples: int = 10000) -> list:
    """Generate naming training data in both languages."""
    examples = []

    print(f"Generating {num_examples:,} naming training examples...")

    # Split between Hebrew and English
    he_count = num_examples // 2
    en_count = num_examples - he_count

    he_examples, next_id = generate_naming_examples_he(he_count)
    examples.extend(he_examples)

    en_examples = generate_naming_examples_en(en_count, next_id)
    examples.extend(en_examples)

    return examples


def save_naming_data(examples: list, output_file: str = "naming_training_data.json"):
    """Save naming examples to a dedicated file."""
    print(f"\nSaving {len(examples):,} naming examples...")

    total_words = sum(len(item["text"].split()) for item in examples)

    data = {
        "metadata": {
            "name": "Creative Naming Training Dataset",
            "version": "1.0",
            "description": "Training data for creative business naming suggestions",
            "language": "he+en",
            "generated_at": datetime.now().isoformat(),
            "num_items": len(examples),
            "total_words": total_words,
        },
        "articles": examples
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(examples):,} naming examples to {output_file}")
    print(f"  Total words: {total_words:,}")

    import os
    print(f"  File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    examples = generate_naming_data(10000)
    save_naming_data(examples)
