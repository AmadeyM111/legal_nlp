#!/usr/bin/env python3
# full_legal_scraper.py
# Полный сбор всех кодексов РФ (НК, ТК, ГК, ЖК, КоАП) с безопасным RPM 10–15 сек
# Один запуск → 5 чистых JSON в ./data/raw/

import json
import random
import re
import time
import unicodedata
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==============================
# Настройки безопасные
# ==============================
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DELAY = 10.0   # секунды
MAX_DELAY = 15.0   # секунды  → ~4–6 запросов в минуту

ua = UserAgent(browsers=["chrome", "firefox"], os=["windows", "macos"])
session = requests.Session()
session.verify = True
session.headers.update({"Accept-Language": "ru-RU,ru;q=0.9"})

_rate_lock = Lock()
_last_request_time = 0.0

# ==============================
# Умная задержка
# ==============================
def respect_delay():
    global _last_request_time
    with _rate_lock:
        sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < sleep_time:
            time.sleep(sleep_time - elapsed)
        _last_request_time = time.time()

# ==============================
# Утилиты
# ==============================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_soup(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript", "aside", "nav", "footer", "iframe"]):
        tag.decompose()
    return soup

# ==============================
# Безопасный запрос
# ==============================
@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, max=30),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def fetch(url: str) -> str:
    respect_delay()
    headers = {"User-Agent": ua.random}
    resp = session.get(url, headers=headers, timeout=40)
    resp.raise_for_status()
    return resp.text

# ==============================
# Стратегии
# ==============================
class ZakonRFStrategy:
    def get_links(self, limit: int) -> List[str]:
        html = fetch(base_url)
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if href.count("/") >= 2 and any(code in href for code in ["/nk/", "/tk/", "/gk/", "/jk/", "/koap/"]):
                full = urljoin(base_url, href)
                if full not in links:
                    links.append(full)
                    if len(links) >= limit:
                        break
        return links

    def parse(self, url: str, html: str) -> Dict[str, Any]:
        soup = clean_soup(BeautifulSoup(html, "html.parser"))
        title = soup.find("h1") or soup.find("title")
        title = normalize_text(title.get_text()) if title else "Без заголовка"
        body = soup.find("div", class_="law-element__body") or soup.find("div", {"id": "law_text_body"})
        content = normalize_text(body.get_text()) if body else ""
        return {"url": url, "title": title, "content": content}

# ==============================
# Основная функция
# ==============================
def scrape_all(limit_per_code: int = 10000):
    CONFIG = {
        "nk":    ("Налоговый кодекс РФ",     "https://www.zakonrf.info/nk/"),
        "tk":    ("Трудовой кодекс РФ",      "https://www.zakonrf.info/tk/"),
        "gk":    ("Гражданский кодекс РФ",   "https://www.zakonrf.info/gk/"),
        "jk":    ("Жилищный кодекс РФ",      "https://www.zakonrf.info/jk/"),
        "koap": ("КоАП РФ",                 "https://www.zakonrf.info/koap/"),
    }

    strategy = ZakonRFStrategy()

    for code, (name, base_url) in CONFIG.items():
        print(f"\n=== Начинаем {name} ({code.upper()}) ==)")
        output_file = OUTPUT_DIR / f"{code}_rf_full.json"

        if output_file.exists():
            print(f"  Уже есть → пропуск")
            continue

        links = strategy.get_links(base_url, limit_per_code)
        print(f"  Найдено ссылок: {len(links)}")

        results = []
        seen = set()

        for i, url in enumerate(links, 1):
            if url in seen:
                continue
            try:
                html = fetch(url)
                article = strategy.parse(url, html)
                if len(article["content"]) > 100:
                    results.append(article)
                    seen.add(url)
                print(f"    [{i}/{len(links)}] {article['title'][:60]}...")
            except Exception as e:
                print(f"    Ошибка {url}: {e}")

        # Сохранение
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"ГОТОВО: {len(results)} статей → {output_file.name}")

    print("\nВсё собрано! 5 кодексов лежат в ./data/raw/")

# ==============================
# Запуск
# ==============================
if __name__ == "__main__":
    scrape_all(limit_per_code=10000)  # 99999 для полного сбора