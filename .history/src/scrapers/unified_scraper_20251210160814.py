#!/usr/bin/env python3

import json
import logging
import random
import re
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# Настройки
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

ua = UserAgent(browsers=["chrome", "firefox"], os=["windows", "macos", "linux"])
SESSION = requests.Session()
SESSION.verify = True  # SSL включён
SESSION.headers.update({"Accept-Language": "ru-RU,ru;q=0.9"})

MAX_WORKERS = 4
RETRY_ATTEMPTS = 5


# ==============================
# Утилиты
# ==============================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_soup(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript", "aside", "nav", "footer"]):
        tag.decompose()
    return soup


# ==============================
# Базовая стратегия
# ==============================
class BaseStrategy:
    def get_article_links(self, base_url: str, limit: int) -> List[str]:
        raise NotImplementedError

    def parse_article(self, url: str, html: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ==============================
# Стратегия: sudact.ru (2025)
# ==============================
class SudactStrategy(BaseStrategy):
    def get_article_links(self, base_url: str, limit: int) -> List[str]:
        links = []
        page = 1
        seen = set()

        while len(links) < limit:
            url = f"{base_url.rstrip('/')}/?page={page}"
            try:
                resp = SESSION.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Sudact оглавление ошибка: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            found = 0
            for a in soup.select("a[href*='/law/']"):
                href = a["href"]
                if "/statya-" in href or "/glava-" in href:
                    full = urljoin("https://sudact.ru", href)
                    if full not in seen:
                        seen.add(full)
                        links.append(full)
                        found += 1
                        if len(links) >= limit:
                            break
            if found == 0:
                break
            page += 1
            time.sleep(random.uniform(1, 3))

        logger.info(f"Sudact: найдено {len(links)} ссылок")
        return links[:limit]

    def parse_article(self, url: str, html: str) -> List[Dict[str, Any]]:
        soup = clean_soup(BeautifulSoup(html, "html.parser"))

        title = soup.find("h1")
        title = normalize_text(title.get_text()) if title else "Без заголовка"

        content = []
        container = soup.find("div", class_="law-text") or soup.find("div", class_="document-text")
        if container:
            for p in container.find_all(["p", "div", "article", "section"], recursive=False):
                txt = normalize_text(p.get_text())
                if txt and len(txt) > 20:
                    content.append(txt)

        return [{"url": url, "title": title, "content": "\n\n".join(content)}]


# ==============================
# Стратегия: zakonrf.info (2025)
# ==============================
class ZakonRFStrategy(BaseStrategy):
    def get_article_links(self, base_url: str, limit: int) -> List[str]:
        # Парсим реальное оглавление вместо угадывания номеров
        links = []
        resp = SESSION.get(base_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.select("a[href^='/'][href*='/']"):
            href = a["href"]
            if href.count("/") >= 2 and any(x in href for x in ["nk/", "gk/", "tk/", "koap/", "jk/"]):
                full = urljoin(base_url, href)
                links.append(full)
                if len(links) >= limit:
                    break

        logger.info(f"ZakonRF: найдено {len(links)} реальных ссылок из оглавления")
        return links[:limit]

    def parse_article(self, url: str, html: str) -> List[Dict[str, Any]]:
        soup = clean_soup(BeautifulSoup(html, "html.parser"))

        title = (
            soup.find("h1", class_="law-element__h1")
            or soup.find("h1")
            or soup.find("title")
        )
        title = normalize_text(title.get_text()) if title else "Без заголовка"

        content_div = soup.find("div", class_="law-element__body") or soup.find("div", id="law_text_body")
        text = normalize_text(content_div.get_text()) if content_div else ""

        return [{"url": url, "title": title, "content": text}]


# ==============================
# Безопасный запрос с retry
# ==============================
@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, max=15),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def fetch_url(url: str) -> str:
    headers = {"User-Agent": ua.random}
    resp = SESSION.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(random.uniform(1.5, 4.0))  # вежливость
    return resp.text


# ==============================
# Основной парсер
# ==============================
def scrape(strategy: BaseStrategy, base_url: str, limit: int, output: Path):
    logger.info(f"Старт парсинга: {strategy.__class__.__name__}")

    links = strategy.get_article_links(base_url, limit)
    if not links:
        logger.error("Ссылки не получены → выход")
        return

    results: List[Dict[str, Any]] = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in links}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            if url in seen_urls:
                continue
            try:
                html = future.result()
                articles = strategy.parse_article(url, html)
                for art in articles:
                    if art["url"] not in seen_urls:
                        seen_urls.add(art["url"])
                        results.append(art)
            except Exception as e:
                logger.error(f"Ошибка {url}: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Готово: {len(results)} статей → {output}")


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    parser = ArgumentParser(description="Скрапер кодексов РФ → чистый JSON (RAG/LoRA)")
    parser.add_argument("code", choices=["nk", "tk", "gk", "jk", "koap"])
    parser.add_argument("--site", choices=["sudact", "zakonrf"], default="zakonrf")
    parser.add_argument("--limit", type=int, default=50, help="Максимум статей")
    parser.add_argument("--output", type=Path, help="Путь к JSON (по умолчанию ./data/raw/)")

    args = parser.parse_args()

    CONFIG = {
        "nk": ("Налоговый кодекс РФ", "https://www.zakonrf.info/nk/", "https://sudact.ru/law/nalogovyi-kodeks-rossiiskoi-federatsii/"),
        "tk": ("Трудовой кодекс РФ", "https://www.zakonrf.info/tk/", "https://sudact.ru/law/trudovoi-kodeks-rossiiskoi-federatsii/"),
        "gk": ("Гражданский кодекс РФ", "https://www.zakonrf.info/gk/", "https://sudact.ru/law/grazhdanskii-kodeks-rossiiskoi-federatsii/"),
        "jk": ("Жилищный кодекс РФ", "https://www.zakonrf.info/jk/", "https://sudact.ru/law/zhilishchnyi-kodeks-rossiiskoi-federatsii/"),
        "koap": ("КоАП РФ", "https://www.zakonrf.info/koap/", "https://sudact.ru/law/kodeks-rossiiskoi-federatsii-ob-administrativnykh-pravonarusheniiakh/"),
    }

    name, url_zakonrf, url_sudact = CONFIG[args.code]
    base_url = url_sudact if args.site == "sudact" else url_zakonrf
    strategy = SudactStrategy() if args.site == "sudact" else ZakonRFStrategy()

    default_out = Path("data/raw") / f"{args.code}_rf_clean.json"
    output_file = args.output or default_out

    print(f"\nПарсим: {name}")
    print(f"Источник: {args.site.upper()} | Лимит: {args.limit} | Выход: {output_file}\n")

    scrape(strategy, base_url, args.limit, output_file)