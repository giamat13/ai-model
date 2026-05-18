"""
collect_corpus.py - איסוף דאטה חיצוני לאימון.

דוגמאות:
    python collect_corpus.py --source wikipedia --out api_training_data.json
    python collect_corpus.py --source github --github-limit 20 --readmes
    python collect_corpus.py --source both --merge-into training_data.json

הערה חשובה:
    GitHub API לא מאפשר "כל ה-repos בעולם" בהרצה אחת. יש rate limits,
    Search API מחזיר עד 1000 תוצאות לשאילתה, והמודל המקומי כאן קטן מדי
    בשביל קורפוס ענק. לכן הסקריפט אוסף מדגם מוגבל ואפשר להריץ אותו שוב עם
    שאילתות שונות. אם מגדירים GITHUB_TOKEN בסביבה, מקבלים מכסת API גבוהה יותר.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_WIKIPEDIA_SEEDS = [
    ("he", "אנגלית"),
    ("he", "ביולוגיה"),
    ("he", "מתמטיקה"),
    ("he", "מדעי המחשב"),
    ("en", "English language"),
    ("en", "Biology"),
    ("en", "Mathematics"),
    ("en", "Computer science"),
]

DEFAULT_GITHUB_QUERIES = [
    "language:Python stars:>1000",
    "topic:machine-learning stars:>1000",
    "topic:mathematics stars:>50",
    "topic:biology stars:>50",
    "topic:education stars:>100",
    "topic:science stars:>100",
]

USER_AGENT = "mini-bilingual-ai-corpus-builder/1.0"


def request_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    req = urllib.request.Request(url, headers=request_headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 429} and attempt < 3:
                wait = 2.0 * (attempt + 1)
                print(f"[HTTP] rate limit זמני, ממתין {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("request failed")


def normalize_text(text: str, max_chars: int) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[#*_>\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def safe_id(prefix: str, *parts: str) -> str:
    raw = "_".join(parts).lower()
    raw = re.sub(r"\W+", "_", raw, flags=re.UNICODE).strip("_")
    return f"{prefix}_{raw}"[:120]


def wikipedia_extract(lang: str, title: str, max_chars: int) -> dict[str, Any] | None:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exintro": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    data = request_json(url)
    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        extract = normalize_text(page.get("extract", ""), max_chars)
        if not extract:
            continue
        resolved_title = page.get("title", title)
        return {
            "id": safe_id("wiki_api", lang, resolved_title),
            "source": f"wikipedia-api:{lang}",
            "topic": resolved_title,
            "language": lang,
            "text": extract,
        }
    return None


def wikipedia_search_titles(lang: str, query: str, limit: int) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
        }
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    data = request_json(url)
    return [
        item["title"]
        for item in data.get("query", {}).get("search", [])
        if item.get("title")
    ]


def wikipedia_random_titles(lang: str, limit: int) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": limit,
        }
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    data = request_json(url)
    return [
        item["title"]
        for item in data.get("query", {}).get("random", [])
        if item.get("title")
    ]


def discover_wikipedia_pages(
    seeds: list[tuple[str, str]], per_seed: int, random_per_lang: int
) -> list[tuple[str, str]]:
    pages: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(lang: str, title: str) -> None:
        key = (lang, title)
        if title and key not in seen:
            seen.add(key)
            pages.append(key)

    for lang, seed in seeds:
        add(lang, seed)
        try:
            for title in wikipedia_search_titles(lang, seed, per_seed):
                add(lang, title)
        except urllib.error.URLError as exc:
            print(f"[Wikipedia] חיפוש נכשל עבור {lang}:{seed} - {exc}")
        time.sleep(0.2)

    for lang in sorted({lang for lang, _ in seeds}):
        try:
            for title in wikipedia_random_titles(lang, random_per_lang):
                add(lang, title)
        except urllib.error.URLError as exc:
            print(f"[Wikipedia] דפי רנדום נכשלו עבור {lang} - {exc}")
        time.sleep(0.2)

    print(f"[Wikipedia] נמצאו אוטומטית {len(pages)} ערכים")
    return pages


def collect_wikipedia(
    pages: list[tuple[str, str]], max_chars: int
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for lang, title in pages:
        try:
            item = wikipedia_extract(lang, title, max_chars)
        except urllib.error.URLError as exc:
            print(f"[Wikipedia] דילוג על {lang}:{title} - {exc}")
            continue
        if item:
            items.append(item)
            print(f"[Wikipedia] נוסף: {lang}:{title}")
        time.sleep(0.4)
    return items


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_search(query: str, limit: int) -> list[dict[str, Any]]:
    per_page = min(max(limit, 1), 100)
    params = urllib.parse.urlencode(
        {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
        }
    )
    url = f"https://api.github.com/search/repositories?{params}"
    data = request_json(url, github_headers())
    return data.get("items", [])[:limit]


def github_readme(full_name: str, max_chars: int) -> str:
    url = f"https://api.github.com/repos/{full_name}/readme"
    data = request_json(url, github_headers())
    content = data.get("content", "")
    if not content:
        return ""
    raw = base64.b64decode(content).decode("utf-8", errors="ignore")
    return normalize_text(raw, max_chars)


def repo_to_text(repo: dict[str, Any], readme: str) -> str:
    parts = [
        f"GitHub repository {repo.get('full_name', '')}.",
        f"Description: {repo.get('description') or 'No description provided'}.",
        f"Primary language: {repo.get('language') or 'unknown'}.",
        f"Stars: {repo.get('stargazers_count', 0)}.",
    ]
    topics = repo.get("topics") or []
    if topics:
        parts.append("Topics: " + ", ".join(topics) + ".")
    if readme:
        parts.append("README excerpt: " + readme)
    return normalize_text(" ".join(parts), 4000)


def collect_github(
    queries: list[str], limit: int, include_readmes: bool, max_chars: int
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for query in queries:
        try:
            repos = github_search(query, limit)
        except urllib.error.URLError as exc:
            print(f"[GitHub] דילוג על שאילתה {query!r} - {exc}")
            continue

        for repo in repos:
            full_name = repo.get("full_name")
            if not full_name or full_name in seen:
                continue
            seen.add(full_name)

            readme = ""
            if include_readmes:
                try:
                    readme = github_readme(full_name, max_chars=2000)
                    time.sleep(0.25)
                except (urllib.error.URLError, KeyError, ValueError) as exc:
                    print(f"[GitHub] README לא נטען עבור {full_name} - {exc}")

            items.append(
                {
                    "id": safe_id("github_api", full_name),
                    "source": "github-api",
                    "topic": full_name,
                    "language": repo.get("language") or "unknown",
                    "text": repo_to_text(repo, readme),
                }
            )
            print(f"[GitHub] נוסף: {full_name}")
        time.sleep(0.5)

    return items


def load_articles(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("articles", [])


def corpus_document(items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "metadata": {
            "name": name,
            "version": "1.0",
            "description": "Generated bilingual corpus from Wikipedia API and GitHub API",
            "language": "he+en+code",
            "generated_at": generated_at,
            "num_items": len(items),
        },
        "articles": items,
    }


def write_corpus(path: str, items: list[dict[str, Any]], name: str) -> None:
    doc = corpus_document(items, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[Write] נשמרו {len(items)} פריטים אל {path}")


def merge_into(path: str, new_items: list[dict[str, Any]]) -> None:
    existing = load_articles(path)
    by_id = {item.get("id"): item for item in existing if item.get("id")}
    for item in new_items:
        by_id[item["id"]] = item

    merged = list(by_id.values())
    doc = corpus_document(merged, "mini_hebrew_english_chat_corpus_merged")
    doc["metadata"]["description"] = (
        "Merged local corpus with API-collected bilingual training data"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[Merge] נשמרו {len(merged)} פריטים אל {path}")


def parse_wiki_page(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use lang:title, for example he:ביולוגיה")
    lang, title = value.split(":", 1)
    return lang.strip(), title.strip()


def parse_wiki_seed(value: str) -> tuple[str, str]:
    return parse_wiki_page(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extra training corpus.")
    parser.add_argument("--source", choices=["wikipedia", "github", "both"], default="both")
    parser.add_argument("--out", default="api_training_data.json")
    parser.add_argument("--merge-into", default="")
    parser.add_argument("--wiki-page", action="append", type=parse_wiki_page)
    parser.add_argument("--wiki-seed", action="append", type=parse_wiki_seed)
    parser.add_argument("--wiki-per-seed", type=int, default=4)
    parser.add_argument("--wiki-random", type=int, default=4)
    parser.add_argument("--no-auto-wiki", action="store_true")
    parser.add_argument("--github-query", action="append")
    parser.add_argument("--github-limit", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=2500)
    parser.add_argument("--readmes", action="store_true")
    args = parser.parse_args()

    items: list[dict[str, Any]] = []
    if args.source in {"wikipedia", "both"}:
        pages = list(args.wiki_page or [])
        if not args.no_auto_wiki:
            seeds = args.wiki_seed or DEFAULT_WIKIPEDIA_SEEDS
            pages.extend(
                discover_wikipedia_pages(
                    seeds,
                    per_seed=max(args.wiki_per_seed, 0),
                    random_per_lang=max(args.wiki_random, 0),
                )
            )
        items.extend(collect_wikipedia(pages, args.max_chars))

    if args.source in {"github", "both"}:
        queries = args.github_query or DEFAULT_GITHUB_QUERIES
        items.extend(collect_github(queries, args.github_limit, args.readmes, args.max_chars))

    if args.merge_into:
        merge_into(args.merge_into, items)
    else:
        write_corpus(args.out, items, "api_training_data")


if __name__ == "__main__":
    main()
