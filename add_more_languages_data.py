"""
add_more_languages_data.py — דוגמאות קוד ושמות שנכתבו ידנית, לא נוצרו אוטומטית.

בניגוד ל-generate_100k_training_data.py (שבוחר random.choice מתוך מאגר קטן
של ~10 תבניות קוד ומשכפל אותו אלפי פעמים תחת ids שונים — הרבה "כמות",
מעט תוכן שונה בפועל), כל דוגמה כאן נכתבה ידנית ומייצגת קוד/שם אמיתי ושונה.
המטרה: איכות על פני נפח. אין כאן לולאות שמגרילות אלפי דוגמאות.

מוסיף:
  - קוד ב-JavaScript, Java, C, C++, Go, SQL, HTML/CSS, Bash — לא רק Python.
  - שמות יצירתיים לקטגוריות חדשות (סטארט-אפ, אפליקציה, להקה, חיית מחמד).
"""

from __future__ import annotations

from data_utils import merge_into_training_data

# כל איבר: (תיאור עברית, תיאור אנגלית, שם שפה ל-```lang, קוד)
CODE_EXAMPLES = [
    ("פונקציה שמדפיסה שלום ב-JavaScript", "a function that prints hello in JavaScript",
     "javascript", "function hello() {\n  console.log('Hello');\n}"),
    ("לולאת for ב-JavaScript שסופרת עד 10", "a JavaScript for loop counting to 10",
     "javascript", "for (let i = 1; i <= 10; i++) {\n  console.log(i);\n}"),
    ("פונקציית חיבור של שני מספרים ב-JavaScript", "a JavaScript function that adds two numbers",
     "javascript", "function add(a, b) {\n  return a + b;\n}"),
    ("בדיקה אם מערך ריק ב-JavaScript", "check if an array is empty in JavaScript",
     "javascript", "function isEmpty(arr) {\n  return arr.length === 0;\n}"),
    ("בקשת fetch ל-API ב-JavaScript", "a fetch request to an API in JavaScript",
     "javascript", "fetch('https://api.example.com/data')\n  .then(res => res.json())\n  .then(data => console.log(data));"),

    ("מחלקת Point ב-Java", "a Point class in Java",
     "java", "class Point {\n    int x, y;\n    Point(int x, int y) {\n        this.x = x;\n        this.y = y;\n    }\n}"),
    ("פונקציה שבודקת אם מספר זוגי ב-Java", "a Java method that checks if a number is even",
     "java", "static boolean isEven(int n) {\n    return n % 2 == 0;\n}"),
    ("לולאת for-each ב-Java", "a Java for-each loop over a list",
     "java", "for (String item : items) {\n    System.out.println(item);\n}"),
    ("Hello World ב-Java", "a Hello World program in Java",
     "java", "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"),

    ("פונקציה שמחשבת עצרת ב-C", "a C function that computes factorial",
     "c", "int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
    ("הדפסת מערך ב-C", "printing an array in C",
     "c", "for (int i = 0; i < n; i++) {\n    printf(\"%d \", arr[i]);\n}"),
    ("החלפת שני משתנים ב-C", "swapping two variables in C",
     "c", "int temp = a;\na = b;\nb = temp;"),

    ("מחלקת Vector2D ב-C++", "a Vector2D class in C++",
     "cpp", "class Vector2D {\npublic:\n    double x, y;\n    Vector2D(double x, double y) : x(x), y(y) {}\n};"),
    ("מיון וקטור ב-C++", "sorting a vector in C++",
     "cpp", "#include <algorithm>\nstd::sort(v.begin(), v.end());"),

    ("פונקציה שבודקת אם מספר ראשוני ב-Go", "a Go function that checks if a number is prime",
     "go", "func isPrime(n int) bool {\n    if n < 2 {\n        return false\n    }\n    for i := 2; i*i <= n; i++ {\n        if n%i == 0 {\n            return false\n        }\n    }\n    return true\n}"),
    ("Hello World ב-Go", "a Hello World program in Go",
     "go", "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}"),

    ("שאילתת SQL שמביאה את כל המשתמשים", "an SQL query that selects all users",
     "sql", "SELECT * FROM users;"),
    ("שאילתת SQL עם JOIN בין הזמנות ללקוחות", "an SQL query joining orders and customers",
     "sql", "SELECT orders.id, customers.name\nFROM orders\nJOIN customers ON orders.customer_id = customers.id;"),
    ("שאילתת SQL שסופרת שורות לפי קבוצה", "an SQL query counting rows grouped by category",
     "sql", "SELECT category, COUNT(*) AS total\nFROM products\nGROUP BY category;"),

    ("דף HTML בסיסי", "a basic HTML page",
     "html", "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n  <h1>Hello</h1>\n</body>\n</html>"),
    ("כפתור עם CSS ב-hover", "a button with a CSS hover effect",
     "css", "button:hover {\n  background-color: #4CAF50;\n  color: white;\n}"),

    ("סקריפט Bash שעובר על קבצים בתיקייה", "a Bash script that loops over files in a directory",
     "bash", "for f in *.txt; do\n  echo \"$f\"\ndone"),
    ("סקריפט Bash שבודק אם קובץ קיים", "a Bash script that checks if a file exists",
     "bash", "if [ -f \"$1\" ]; then\n  echo \"exists\"\nfi"),
]


def build_code_examples() -> list[dict]:
    examples = []
    for i, (desc_he, desc_en, lang, code) in enumerate(CODE_EXAMPLES):
        text_he = f"User: כתוב קוד שיעשה {desc_he} Model: ```{lang}\n{code}\n```"
        text_en = f"User: Write code to {desc_en} Model: ```{lang}\n{code}\n```"
        examples.append({
            "id": f"code_lang_he_{i:04d}",
            "source": "manual_more_languages",
            "language": "he",
            "topic": "programming",
            "intent": "coding",
            "text": text_he,
        })
        examples.append({
            "id": f"code_lang_en_{i:04d}",
            "source": "manual_more_languages",
            "language": "en",
            "topic": "programming",
            "intent": "coding",
            "text": text_en,
        })
    return examples


# כל איבר: (קטגוריה עברית, קטגוריה אנגלית, רשימת שמות עברית, רשימת שמות אנגלית)
NAMING_EXAMPLES = [
    ("סטארט-אפ טכנולוגי", "a tech startup",
     ["נקודת אור", "קוד פתוח", "המצאה", "צעד קדימה", "חדשנות שקטה"],
     ["Nextwave", "Brightloop", "Corevine", "Pathforge", "Quietspark"]),
    ("אפליקציית מובייל לניהול זמן", "a mobile app for time management",
     ["רגע שלי", "הזמן שלך", "לוח זמנים חכם", "יום מסודר", "קצב"],
     ["Timely", "Dayflow", "Momently", "ClearHours", "Pacer"]),
    ("להקת רוק", "a rock band",
     ["האש השחורה", "רעם רחוק", "הקול האבוד", "מסע הברזל", "צל הלילה"],
     ["Iron Echo", "Distant Thunder", "Lost Static", "Night Shadow", "Rust & Fire"]),
    ("כלב חמוד", "a cute dog",
     ["פלאפל", "ברקו", "צ'ופי", "שמש", "קטיפה"],
     ["Biscuit", "Nugget", "Waffles", "Sunny", "Pepper"]),
    ("חתול", "a cat",
     ["פרווה", "נמר קטן", "ליצן", "צל", "שקד"],
     ["Whiskers", "Shadow", "Tiger", "Marshmallow", "Clover"]),
]


def build_naming_examples() -> list[dict]:
    examples = []
    for i, (cat_he, cat_en, names_he, names_en) in enumerate(NAMING_EXAMPLES):
        names_he_str = " . ".join(names_he)
        names_en_str = " . ".join(names_en)
        examples.append({
            "id": f"naming_more_he_{i:04d}",
            "source": "manual_more_naming",
            "language": "he",
            "type": "naming",
            "text": f"User: תן שמות יצירתיים ל{cat_he} Model: הנה רעיונות : {names_he_str} .",
        })
        examples.append({
            "id": f"naming_more_en_{i:04d}",
            "source": "manual_more_naming",
            "language": "en",
            "type": "naming",
            "text": f"User: give creative names for {cat_en} Model: Here are some ideas : {names_en_str} .",
        })
    return examples


if __name__ == "__main__":
    examples = build_code_examples() + build_naming_examples()
    merge_into_training_data(examples, "MoreLanguagesAndNaming")
