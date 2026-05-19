"""
fetch_articles.py - מוריד תוכן ממאמרים לפי קישורים ב-article_links.json

שימוש:
    python fetch_articles.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
LINKS_FILE = os.path.join(_HERE, "article_links.json")
OUTPUT_FILE = os.path.join(_HERE, "fetched_articles.json")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class TextExtractor(HTMLParser):
    """מחלץ טקסט מ-HTML"""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
    
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip = True
    
    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip = False
    
    def handle_data(self, data):
        if not self.skip:
            self.text.append(data.strip())
    
    def get_text(self):
        return " ".join(t for t in self.text if t)


def fetch_url(url: str) -> str:
    """מוריד תוכן מ-URL"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
            parser = TextExtractor()
            parser.feed(html)
            return parser.get_text()
    except Exception as e:
        print(f"[Error] לא הצלחתי להוריד {url}: {e}")
        return ""


def load_links() -> list[dict]:
    """טוען קישורים מ-article_links.json"""
    if not os.path.exists(LINKS_FILE):
        print(f"[Error] קובץ {LINKS_FILE} לא נמצא")
        return []
    
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("links", [])


def save_articles(articles: list[dict]) -> None:
    """שומר מאמרים ל-JSON"""
    output = {
        "metadata": {
            "name": "fetched_articles",
            "description": "מאמרים שהורדו מקישורים",
            "version": "1.0",
            "num_items": len(articles)
        },
        "articles": articles
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[Save] נשמרו {len(articles)} מאמרים ל-{OUTPUT_FILE}")


def main():
    print("=" * 56)
    print("  📥  הורדת מאמרים מקישורים")
    print("=" * 56)
    
    links = load_links()
    if not links:
        print("[Info] אין קישורים להורדה. הוסף קישורים ל-article_links.json")
        return
    
    articles = []
    for i, link_data in enumerate(links, 1):
        url = link_data.get("url", "")
        title = link_data.get("title", f"article_{i}")
        
        if not url:
            continue
        
        print(f"[{i}/{len(links)}] מוריד: {title}")
        text = fetch_url(url)
        
        if text:
            articles.append({
                "id": f"fetched_{i}_{title.replace(' ', '_')}",
                "text": text[:5000],  # מגביל ל-5000 תווים
                "source": "fetched_article",
                "topic": title,
                "url": url
            })
            print(f"  ✓ הורד {len(text)} תווים")
        
        time.sleep(1)  # המתנה בין בקשות
    
    if articles:
        save_articles(articles)
        print(f"\n✅ הסתיים! {len(articles)} מאמרים נשמרו")
    else:
        print("\n⚠️ לא הורדו מאמרים")


if __name__ == "__main__":
    main()
