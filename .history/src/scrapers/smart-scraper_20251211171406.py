#!/usr/bin/env python3
# smart_legal_scraper.py
# 150 лучших статей на кодекс с идеальным покрытием разделов
# Выгрузка → ./data/rag_ready/{nk,gk,tk,jk,koap}_150.json

import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# Корневой каталог проекта — работает из любой папки
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CODE = 150
MIN_CONTENT_LENGTH = 800

def fetch(url: str) -> str:
    for _ in range(5):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            time.sleep(random.uniform(2.5, 5.0))  # сверхвежливый режим
            return r.text
        except:
            time.sleep(10)
    return ""

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Главная магия — берём именно оглавление и равномерно по главам
CONFIG = {
    "nk":   "https://www.zakonrf.info/nk/",
    "gk":   "https://www.zakonrf.info/gk/",
    "tk":   "https://www.zakonrf.info/tk/",
    "jk":   "https://www.zakonrf.info/jk/",
    "koap": "https://www.zakonrf.info/koap/",
}

def get_balanced_articles(code: str, base_url: str):
    print(f"Собираем [{code.upper()}] — 150 статей с балансом по разделам...")
    html = fetch(base_url)
    soup = BeautifulSoup(html, "html.parser")

    # Ищем все ссылки на главы/разделы/части
    chapter_links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if any(keyword in text for keyword in ["часть", "раздел", "глава", "том"]) or href.count("/") >= 2:
            full = urljoin(base_url, href)
            if full.startswith(base_url) and full != base_url:
                chapter_links.append(full)

    chapter_links = list(dict.fromkeys(chapter_links))[:30]  # максимум 30 глав

    if not chapter_links:
        print(f" Оглавление не найдено - используум прямой перебор статей по номерам")
        chapter_links = [base_url]

    n_chapters = max(1, len(chapter_links))  # никогда не будет 0
    articles_per_chapter = max(1, TARGET_PER_CODE // n_chapters)
    result = []
    seen_urls = set()

    for chapter_url in chapter_links:
        print(f"  Обрабатываем раздел: {chapter_url.split('/')[-2]}")
        html = fetch(chapter_url)
        soup = BeautifulSoup(html, "html.parser")
        links = soup.select("a[href]")
        candidates = []

        for a in links:
            href = a["href"]
            if "/"+str(code)+"/" in href and href.endswith("/"):
                num = href.strip("/").split("/")[-1]
                if num.isdigit():
                    url = urljoin("https://www.zakonrf.info", href)
                    if url not in seen_urls:
                        candidates.append(url)

        # Берём равномерно из главы
        step = max(1, len(candidates) // articles_per_chapter)
        selected = candidates[::step][:articles_per_chapter]

        for url in selected:
            if len(result) >= TARGET_PER_CODE:
                break
            html = fetch(url)
            soup = BeautifulSoup(html, "html.parser")
            title = soup.find("h1")
            title = clean_text(title.get_text()) if title else "Без заголовка"

            body = soup.find("div", class_="law-element__body") or soup.find("div", id="law_text_body")
            content = clean_text(body.get_text(separator="\n")) if body else ""

            if len(content) > MIN_CONTENT_LENGTH and url not in seen_urls:
                result.append({"url": url, "title": title, "content": content})
                seen_urls.add(url)
                print(f"    +", title[:60], f"({len(content)} симв.)")

    return result[:TARGET_PER_CODE]

# Запуск
if __name__ == "__main__":
    for code, url in CONFIG.items():
        articles = get_balanced_articles(code, url)
        out_file = OUTPUT_DIR / f"{code}_150.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"{code.upper()} — {len(articles)} статей → {out_file.name}\n")

    print("Готово! 750 лучших статей (по 150 на кодекс) в ./data/rag_ready/")