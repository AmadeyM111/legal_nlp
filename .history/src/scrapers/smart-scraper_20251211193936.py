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
from functools import lru_cache

@lru_cache(maxsize=128)
def get_all_links_cached(base_url: str) -> list[str]:
    return ZakonRFStrategy().get_links(base_url)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

url = f"{base_url}/{article_id}/"

TARGET_PER_CODE = 10
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

def get_balanced_articles(code: str, base_url: str) -> list[dict]:
    logger.info(f"[{code.upper()}] Сбор {TARGET_PER_CODE} статей с балансом по разделам...")
    
    result = []
    seen_urls = set()

    # 1. Получаем главную страницу
    html = fetch(base_url)
    if not html:
        logger.error(f"[{code.upper()}] Не удалось загрузить главную страницу")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # 2. Ищем ссылки на главы
    chapter_links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        full_url = urljoin(base_url, href)
        
        if any(keyword in text for keyword in ["часть", "раздел", "глава", "том"]) or href.count("/") >= 2:
            full = urljoin(base_url, href)
            if full.startswith(base_url) and full != base_url:
                chapter_links.append(full)
                seen_urls.add(full_url)

    chapter_links = list(dict.fromkeys(chapter_links))[:30]  # максимум 30 глав
    if not chapter_links:
        logger.warrning(f"[{code.upper()}] Оглавление не найдено - перебор по номерам")
        chapter_links = [base_url]

    article_per_chapter = max(1, target // len(chapter_links))

    for chapter_url in chapter_links:
        if len(result) >= target:
            break 

        logger.info(f"  ")
        print(f"  Обрабатываем раздел: {chapter_url.split('/')[-2]}")
        html = fetch(chapter_url)
        soup = BeautifulSoup(html, "html.parser")
        links = soup.select("a[href]")
        candidates = []

        for a in soup.select("a[href]"):
            href = a["href"]
            full = urljoin("https://www.zakonrf.info", href)
            if re.search(rf"/{code}/\d+/?$", full) and full not in seen_urls:
                candidates.append(full)

        # Берём равномерно из главы
        step = max(1, len(candidates) // articles_per_chapter)
        selected = candidates[::step][:articles_per_chapter + 5]

        for url in selected:
            if len(result) >= TARGET_PER_CODE:
                break
            article = parse_article_safely(url)
            if article and url not in seen_urls:
                result.append(article)
                seen_urls.add(url)
                print(f"    + {article['title'][:60]}...({len(article['content'])} симв.)")
                return result

            if len(content) > MIN_CONTENT_LENGTH and url not in seen_urls:
                result.append({"base_url": url, "title": title, "content": content})
                seen_urls.add(url)
                print(f"    +", title[:60], f"({len(content)} симв.)")

    return result[:TARGET_PER_CODE]

# Запуск
if __name__ == "__main__":

    CONFIG = {
    "nk":   "https://www.zakonrf.info/nk/",
    "gk":   "https://www.zakonrf.info/gk/",
    "tk":   "https://www.zakonrf.info/tk/",
    "jk":   "https://www.zakonrf.info/jk/",
    "koap": "https://www.zakonrf.info/koap/",
}
    
    total = 0
    for code, base_url in CONFIG.items():
        print(f"\n{'='*60}")
        print(f"ЗАПУСК: {code.upper()} - {base_url}")
        articles = get_balanced_articles(code, url)

        out_file = OUTPUT_DIR / f"{code}_150.json"
        with open(out_file, "w", encoding="utf-8") as f:            
            json.dump(articles, f, ensure_ascii=False, indent=True)

        print(f"{code.upper()} — {len(articles)} статей → {out_file.name}\n")
        total += len(articles)

    print(f"\n{'='*60}")
    print(f"ГОТОВО! Собрано {total} статей (5 кодексов)")
    print(f"Папка: ./{OUTPUT_DIR}/")