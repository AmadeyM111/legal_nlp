#!/usr/bin/env python3
"""
Сбор 150 лучших статей по каждому кодексу с балансом по разделам
"""

import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# === Конфигурация ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CODE = 150
MIN_CONTENT_LENGTH = 800

# Глобальная сессия (переиспользуется)
session = requests.Session()
session.headers.update({"User-Agent": UserAgent().random})

# === Вспомогательные функции ===
def fetch(url: str) -> str:
    for attempt in range(5):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(random.uniform(2.5, 5.0))
            return response.text
        except Exception as e:
            print(f"    Попытка {attempt+1} не удалась: {e}")
            time.sleep(10)
    return ""

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def parse_article_safely(url: str) -> dict | None:
    html = fetch(url)
    if not html or "404" in html or "не найдена" in html.lower():
        return None

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    body = soup.find("div", class_="law-element__body") or soup.find("div", id="law_text_body")

    if not title or not body:
        return None

    title_text = clean_text(title.get_text())
    content = clean_text(body.get_text(separator="\n"))

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {"url": url, "title": title_text, "content": content}

# === Основная логика ===
def get_balanced_articles(code: str, base_url: str, target: int = TARGET_PER_CODE) -> list[dict]:
    print(f"[{code.upper()}] Сбор {target} статей с балансом по разделам...")

    result = []
    seen_urls = set()

    # 1. Главная страница
    html = fetch(base_url)
    if not html:
        print(f"[{code.upper()}] Не удалось загрузить {base_url}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # 2. Ищем главы
    chapter_links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        full = urljoin(base_url, href)

        if (any(kw in text for kw in ["часть", "раздел", "глава", "том"]) or
            href.count("/") >= 2) and full.startswith(base_url) and full != base_url:
            if full not in seen_urls:
                chapter_links.append(full)
                seen_urls.add(full)

    chapter_links = list(dict.fromkeys(chapter_links))[:30]

        if not chapter_links:
        print(f"  Оглавление не найдено — используем перебор")
            chapter_links = [base_url]

    articles_per_chapter = max(1, target // len(chapter_links))

    # 3. Сбор статей
    for chapter_url in chapter_links:
        if len(result) >= target:
            break

        print(f"  Раздел: {chapter_url.split('/')[-2] if '/' in chapter_url else 'главная'}")
        html = fetch(chapter_url)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        candidates = []

        for a in soup.select("a[href]"):
            href = a["href"]
            full = urljoin("https://www.zakonrf.info", href)
            if re.search(rf"/{code}/\d+/?$", full) and full not in seen_urls:
                candidates.append(full)

        step = max(1, len(candidates) // articles_per_chapter)
        selected = candidates[::step][:articles_per_chapter]

        for url in selected:
            if len(result) >= target:
                break
            article = parse_article_safely(url)
            if article:
                result.append(article)
                seen_urls.add(url)
                print(f"    + {article['title'][:60]}... ({len(article['content'])} симв.)")

    print(f"[{code.upper()}] Готово: {len(result)} статей")
    return result

# === Запуск ===
if __name__ == "__main__":
    CONFIG = {
        "nk":    "https://www.zakonrf.info/nk/",
        "tk":    "https://www.zakonrf.info/tk/",
        "gk":    "https://www.zakonrf.info/gk/",
        "jk":    "https://www.zakonrf.info/jk/",
        "koap":  "https://www.zakonrf.info/koap/",
    }

    total = 0
    for code, base_url in CONFIG.items():
        print(f"\n{'='*60}")
        print(f"ЗАПУСК: {code.upper()}")
        articles = get_balanced_articles(code, base_url)

        out_file = OUTPUT_DIR / f"{code}_150.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"{code.upper()} — {len(articles)} статей → {out_file.name}")
        total += len(articles)

    print(f"\n{'='*60}")
    print(f"ГОТОВО! Собрано {total} статей")
    print(f"Папка: {OUTPUT_DIR}/")