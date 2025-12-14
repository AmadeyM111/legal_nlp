#!/usr/bin/env python3
import json
import random
import re
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# -----------------------------
# Config
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/Users/antonamadeus/github-projects/Active/experimental")
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CODE = 150
MIN_CONTENT_LENGTH = 200  # для "официального текста статьи" 800 часто слишком много

CODES = ["nk", "gk", "tk", "jk", "koap"]
BASE = "https://www.zakonrf.info"

STOP_LINES = {"Закрыть", "Развернуть", "Свернуть"}


# -----------------------------
# Session
# -----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
})


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException,))
)
def fetch(url: str, timeout: int = 30) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_article_number(title: str) -> Optional[str]:
    """
    Извлекает номер статьи из заголовка:
    "Статья 2.3 КоАП РФ...." -> "2.3"
    """
    if not title:
        return None
    m = re.search(r"Статья\s+(\d+(?:\.\d+)*)", title)
    return m.group(1) if m else None


def is_zakonrf_article_page(soup: BeautifulSoup) -> bool:
    """
    Страница статьи: есть h1 в article header и есть хотя бы 1 p в article>div
    """
    if not soup.select_one("article header h1"):
        return False
    return len(soup.select("article > div p")) > 0


def parse_zakonrf_article(url: str, code: str) -> Optional[Dict]:
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("article header h1") or soup.select_one("h1")
    if not h1:
        return None

    title = clean_text(h1.get_text(" ", strip=True))
    article_number = extract_article_number(title)

    container = soup.select_one("article > div") or soup.select_one("article")
    if not container:
        return None

    parts = []
    for p in container.select("p"):
        t = p.get_text(" ", strip=True)
        t = clean_text(t)
        if not t or t in STOP_LINES:
            continue
        if len(t) < 3:
            continue
        parts.append(t)

    content = "\n".join(parts).strip()

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {
        "url": url,
        "title": title,
        "content": content,
        "code": code,
        "article_number": article_number,
        "source": "zakonrf.info",
        "fetch_date": time.strftime("%Y-%m-%d")
    }


def crawl_zakonrf_collect_article_urls(start_url: str, code: str, limit_pages: int = 50) -> list[str]:
    """Рекурсивный краулер для сбора URL статей с zakonrf.info"""
    # статьи: /{code}/123/ или /{code}/2.3/
    re_article = re.compile(rf"^/{code}/\d+(?:\.\d+)*\/?$")

    q = deque([start_url])
    seen_pages = set([start_url])
    found_articles = set()

    while q and len(seen_pages) <= limit_pages and len(found_articles) < 5000:
        url = q.popleft()
        html = fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        # 1) ссылки на странице (TOC)
        tree = soup.select_one("ul.law-element__tree")
        anchors = tree.select("a[href]") if tree else soup.select("article a[href], main a[href]")

        for a in anchors:
            full = urljoin(url, a.get("href"))
            path = urlparse(full).path

            if not path.startswith(f"/{code}/"):
                continue

            # статья?
            if re_article.match(path):
                found_articles.add(full if full.endswith("/") else full + "/")
            else:
                # промежуточная страница (глава/часть/раздел)
                if full not in seen_pages:
                    seen_pages.add(full)
                    q.append(full)

        time.sleep(random.uniform(0.4, 1.0))

    return sorted(found_articles)


def parse_zakonrf_index_links(index_url: str, code: str) -> List[str]:
    """
    Для /koap/, /nk/, /gk/... вытаскиваем ссылки из ul.law-element__tree (если есть),
    иначе берём все ссылки внутри article/main как fallback.
    Также используем рекурсивный краулер для полного охвата.

    Важно: фильтруем по пути /{code}/...
    """
    # Используем рекурсивный краулер для полного сбора URL статей
    logger.info(f"[{code.upper()}] Сбор ссылок через рекурсивный краулер...")
    crawled_urls = crawl_zakonrf_collect_article_urls(index_url, code)
    
    # Также используем старый метод как дополнительный источник
    html = fetch(index_url)
    soup = BeautifulSoup(html, "html.parser")

    candidates = []

    tree = soup.select_one("ul.law-element__tree")
    if tree:
        anchors = tree.select("a[href]")
    else:
        anchors = soup.select("article a[href], main a[href]")

    for a in anchors:
        href = a.get("href")
        if not href:
            continue
        full = urljoin(index_url, href)
        path = urlparse(full).path

        if path.startswith(f"/{code}/"):
            candidates.append(full)

    # Объединяем результаты обоих методов
    all_candidates = crawled_urls + candidates

    # уникализация с сохранением порядка
    seen = set()
    out = []
    for u in all_candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def filter_article_urls(urls: List[str], code: str) -> List[str]:
    """
    Отделяем статьи от глав/разделов:
    статьи zakonrf: /{code}/<digits(.digits)*>/
    главы/разделы часто: /{code}/gl1/, /{code}/r1/ и т.п.
    """
    re_article = re.compile(rf"^https://www\.zakonrf\.info/{code}/\d+(?:\.\d+)*\/?$")
    return [u if u.endswith("/") else u + "/" for u in urls if re_article.match(u)]


def scrape_zakonrf_code(code: str, target: int = TARGET_PER_CODE) -> List[Dict]:
    index_url = f"{BASE}/{code}/"
    logger.info(f"[{code.upper()}] index: {index_url}")

    # 1) берём ссылки с оглавления (теперь с рекурсивным краулером)
    all_links = parse_zakonrf_index_links(index_url, code)
    logger.info(f"[{code.upper()}] links from index: {len(all_links)}")

    # 2) фильтруем только статьи
    article_urls = filter_article_urls(all_links, code)
    logger.info(f"[{code.upper()}] article urls: {len(article_urls)}")

    # 3) берём случайную подвыборку (для прототипа)
    random.shuffle(article_urls)
    article_urls = article_urls[: max(target * 2, target)]  # запас на отбраковку

    # 4) парсим статьи
    out = []
    for i, url in enumerate(article_urls, 1):
        if len(out) >= target:
            break

        try:
            art = parse_zakonrf_article(url, code)
        except Exception as e:
            logger.warning(f"[{code.upper()}] error fetch {url}: {e}")
            art = None

        if art:
            out.append(art)
            logger.info(f"[{code.upper()}] + {i}/{len(article_urls)} | {art.get('article_number')} | {art['title'][:70]}")
        else:
            logger.info(f"[{code.upper()}] - {i}/{len(article_urls)} | skipped: {url}")

        time.sleep(random.uniform(0.8, 2.2))

    out = merge_dedupe_by_code_article(out)
    out = out[:target]
    logger.info(f"[{code.upper()}] collected: {len(out)}")
    return out


def merge_dedupe_by_code_article(items: List[Dict]) -> List[Dict]:
    """
    Дедуп по (code, article_number). Если article_number нет — по URL.
    При конфликте оставляем запись с более длинным content.
    """
    best = {}
    for it in items:
        key = (it.get("code"), it.get("article_number")) if it.get("article_number") else ("url", it["url"])
        if key not in best or len(it["content"]) > len(best[key]["content"]):
            best[key] = it
    return list(best.values())


def scrape_zakonrf_code(code: str, target: int = TARGET_PER_CODE) -> List[Dict]:
    index_url = f"{BASE}/{code}/"
    logger.info(f"[{code.upper()}] index: {index_url}")

    # 1) берём ссылки с оглавления
    all_links = parse_zakonrf_index_links(index_url, code)
    logger.info(f"[{code.upper()}] links from index: {len(all_links)}")

    # 2) фильтруем только статьи
    article_urls = filter_article_urls(all_links, code)
    logger.info(f"[{code.upper()}] article urls: {len(article_urls)}")

    # 3) берём случайную подвыборку (для прототипа)
    random.shuffle(article_urls)
    article_urls = article_urls[: max(target * 2, target)]  # запас на отбраковку

    # 4) парсим статьи
    out = []
    for i, url in enumerate(article_urls, 1):
        if len(out) >= target:
            break

        try:
            art = parse_zakonrf_article(url, code)
        except Exception as e:
            logger.warning(f"[{code.upper()}] error fetch {url}: {e}")
            art = None

        if art:
            out.append(art)
            logger.info(f"[{code.upper()}] + {i}/{len(article_urls)} | {art.get('article_number')} | {art['title'][:70]}")
        else:
            logger.info(f"[{code.upper()}] - {i}/{len(article_urls)} | skipped: {url}")

        time.sleep(random.uniform(0.8, 2.2))

    out = merge_dedupe_by_code_article(out)
    out = out[:target]
    logger.info(f"[{code.upper()}] collected: {len(out)}")
    return out


def main():
    all_stats = {}
    for code in CODES:
        logger.info("=" * 70)
        articles = scrape_zakonrf_code(code, TARGET_PER_CODE)
        all_stats[code] = len(articles)

        out_path = OUTPUT_DIR / f"zakonrf_{code}_{len(articles)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"[{code.upper()}] saved -> {out_path}")

    logger.info(f"Done. Stats: {all_stats}")


if __name__ == "__main__":
    main()
