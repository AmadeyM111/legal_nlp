#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import logging
import time
import random
import json
from pathlib import Path
import argparse
import urllib3
from typing import List, Dict, Any

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# Конфигурация кодексов
# ==============================
CONFIG = {
    "tk":   {"name": "Трудовой кодекс РФ",     "url": "https://www.zakonrf.info/tk/{}/",      "sudact": "https://sudact.ru/law/tk-rf/"},
    "gk":   {"name": "Гражданский кодекс РФ",  "url": "https://www.zakonrf.info/gk/{}/",      "sudact": "https://sudact.ru/law/gk-rf/"},
    "nk":   {"name": "Налоговый кодекс РФ",    "url": "https://www.zakonrf.info/nk/{}/",      "sudact": "https://sudact.ru/law/nk-rf/"},
    "jk":   {"name": "Жилищный кодекс РФ",     "url": "https://www.zakonrf.info/jk/{}/",      "sudact": "https://sudact.ru/law/jk-rf/"},
    "koap": {"name": "КоАП РФ",                "url": "https://www.zakonrf.info/koap/{}/",    "sudact": "https://sudact.ru/law/koap-rf/"},
}

# ==============================
# Парсер (zakonrf.info — стабильнее всего)
# ==============================
def fetch_article(url: str) -> Dict[str, Any]:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=20, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Title
        title = soup.find("h1", class_="law-element__h1")
        title = title.get_text(strip=True) if title else soup.title.get_text(strip=True)

        # Content
        body = soup.find("div", class_="law-element__body") or soup.find("div", id="law_text_body")
        if body:
            for bad in body(["script", "style", "aside", "pre"]):
                bad.decompose()
            content = body.get_text(separator="\n", strip=True)
        else:
            content = "Content not found"

        return {"url": url, "title": title, "content": content}
    except Exception as e:
        logger.error(f"Ошибка {url}: {e}")
        return None


# ==============================
# Генерация всех трёх датасетов за один проход
# ==============================
def export_all_datasets(code: str, articles: List[Dict], max_articles: int):
    code_dir = Path(f"data/{code}")
    raw_dir = code_dir / "raw"
    proc_dir = code_dir / "processed"
    dist_dir = code_dir / "distilled"

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / f"{code}_full.jsonl"
    clean_file = proc_dir / f"{code}_cases.jsonl"

    with open(raw_file, 'w', encoding='utf-8') as raw_f, \
         open(clean_file, 'w', encoding='utf-8') as clean_f:

        for art in articles[:max_articles]:
            if not art:
                continue

            # 1. Raw — всё как есть
            json.dump(art, raw_f, ensure_ascii=False)
            raw_f.write('\n')

            # 2. Clean — {"case": "...", "article": "72"}
            try:
                article_num = art["title"].split("Статья")[1].split()[0].split(".")[0].strip()
                article_num = article_num.replace("№", "").strip()
            except:
                article_num = "unknown"

            # Убираем номер статьи из текста
            clean_text = art["content"]
            for prefix in [f"Статья {article_num}", f"{article_num}.", f"Ст.{article_num}"]:
                clean_text = clean_text.replace(prefix, "").strip()

            case = clean_text[:1500] + "..." if len(clean_text) > 1500 else clean_text

            clean_item = {"case": case.strip(), "article": article_num}
            json.dump(clean_item, clean_f, ensure_ascii=False)
            clean_f.write('\n')

            # 3. Distilled — заглушка (потом заполнишь через Grok-4/GPT-4o)
            # Пример: одна статья → 20 синтетических кейсов
            # Запусти отдельно скрипт distill.py

    logger.info(f"✓ {code.upper()} | Raw: {raw_file} | Clean: {clean_file} | Готово к дистилляции → {dist_dir}")


# ==============================
# Основной цикл
# ==============================
def scrape_code(code: str, limit: int = 400):
    if code not in CONFIG:
        raise ValueError(f"Кодекс {code} не поддерживается")

    base_url = CONFIG[code]["url"]
    logger.info(f"Скрапим {CONFIG[code]['name']} — до {limit} статей")

    articles = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})

    for i in range(1, limit + 1):
        url = base_url.format(i)
        print(f"[{i}/{limit}] {url}")
        art = fetch_article(url)
        if art:
            articles.append(art)
            print(f"  ✓ {art['title'][:70]}")
        else:
            print("  ✗ пропущено")
        time.sleep(random.uniform(0.7, 1.8))

    export_all_datasets(code, articles, limit)


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрапер кодексов РФ → полный пайплайн ML")
    parser.add_argument("code", choices=["tk", "gk", "nk", "jk", "koap"])
    parser.add_argument("--limit", type=int, default=400, help="Макс. статей (ТК — 424, ГК — 1500+)")
    args = parser.parse_args()

    scrape_code(args.code, args.limit)