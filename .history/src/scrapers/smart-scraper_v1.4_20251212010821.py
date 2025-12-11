#!/usr/bin/env python3
# smart_legal_scraper.py
# 100 статей на кодекс с идеальным покрытием разделов
# Выгрузка → ./data/raw/{nk,gk,tk,jk,koap}_100.json

import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import logging

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Корневой каталог проекта — работает из любой папки
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CODE = 100  # Изменено на 100 статей
MIN_CONTENT_LENGTH = 800

# Создаем сессию с User-Agent
ua = UserAgent()
session = requests.Session()
session.headers.update({
    'User-Agent': ua.random,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'TE': 'trailers',
})

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
)
def fetch(url: str) -> str:
    """Загрузка страницы с обработкой ошибок и повторными попытками"""
    logger.info(f"Загрузка: {url}")
    response = session.get(url, timeout=30)
    response.raise_for_status()
    logger.info(f"Успешно загружено: {url}")
    return response.text

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, AttributeError))
)
def parse_article_safely(url: str) -> dict:
    """Безопасное извлечение статьи с обработкой ошибок и повторными попытками"""
    logger.info(f"Парсинг статьи: {url}")
    
    html = fetch(url)
    if not html:
        logger.warning(f"Не удалось получить HTML для {url}")
        return None
        
    soup = BeautifulSoup(html, "html.parser")
    
    title_tag = soup.find("h1")
    body = soup.find("div", class_="law-element__body") or soup.find("div", id="law_text_body")

    if not title_tag or not body:
        logger.warning(f"Не найдены заголовок или тело статьи для {url}")
        return None

    title = clean_text(title_tag.get_text())
    content = clean_text(body.get_text(separator="\n"))

    if len(content) < MIN_CONTENT_LENGTH:
        logger.info(f"Контент слишком короткий для {url}: {len(content)} символов")
        return None

    article = {"url": url, "title": title, "content": content}
    logger.info(f"Статья успешно извлечена: {title[:50]}...")
    return article

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

    # 2. Ищем ссылки на главы/разделы
    chapter_links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = a.get_text(strip=True).lower()
        full_url = urljoin(base_url, href)
        
        # Проверяем, является ли ссылка ссылкой на главу/раздел
        if (any(keyword in text for keyword in ["часть", "раздел", "глава", "том"]) 
            or href.count("/") >= 2) and full_url.startswith(base_url) and full_url != base_url:
            if full_url not in seen_urls:
                chapter_links.append(full_url)
                seen_urls.add(full_url)

    chapter_links = list(dict.fromkeys(chapter_links))[:30]  # максимум 30 глав
    
    if not chapter_links:
        logger.info(f"    Оглавление не найдено - перебор по номерам")
        chapter_links = [base_url]

    articles_per_chapter = max(1, TARGET_PER_CODE // len(chapter_links))

    for chapter_url in chapter_links:
        if len(result) >= TARGET_PER_CODE:
            break 

        logger.info(f"  Обрабатываем раздел: {chapter_url.split('/')[-2] if '/' in chapter_url else 'главная'}")
        html = fetch(chapter_url)
        if not html: 
            continue

        soup = BeautifulSoup(html, "html.parser")
        
        # Ищем ссылки на статьи в текущем разделе
        candidates = []
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            full_url = urljoin("https://www.zakonrf.info", href)
            # Проверяем, что это ссылка на статью в текущем кодексе
            if re.search(rf"/{code}/\d+/?$", href) and full_url not in seen_urls:
                candidates.append(full_url)

        # Берём равномерно из главы
        step = max(1, len(candidates) // articles_per_chapter)
        selected = candidates[::step][:articles_per_chapter + 5]

        for url in selected:
            if len(result) >= TARGET_PER_CODE:
                break
            article = parse_article_safely(url)
            if article and article['url'] not in seen_urls:
                result.append(article)
                seen_urls.add(article['url'])
                logger.info(f"    + {article['title'][:60]}...({len(article['content'])} симв.)")
                
                # Задержка между успешными запросами для избегания антибот-проверки
                delay = random.uniform(3.0, 7.0)
                logger.info(f"Задержка {delay:.2f} секунд между запросами")
                time.sleep(delay)
            
            if len(result) >= TARGET_PER_CODE:
                break

    # Если по оглавлению не удалось собрать нужное количество статей, 
    # используем перебор по номерам
    if len(result) < TARGET_PER_CODE:
        logger.info(f"  Дозагрузка перебором по номерам...")
        article_id = 1
        attempts_without_success = 0
        max_empty_attempts = 50  # Увеличено для большего охвата

        while len(result) < TARGET_PER_CODE and attempts_without_success < max_empty_attempts:
            url = f"{base_url.rstrip('/')}/{article_id}/"
            
            article = parse_article_safely(url)
            
            if article and article['url'] not in seen_urls:
                result.append(article)
                seen_urls.add(article['url'])
                logger.info(f"  + {article['title'][:70]} ({len(article['content'])} симв.)")
                attempts_without_success = 0  # сбрасываем счетчик при успехе
                
                # Задержка между успешными запросами
                delay = random.uniform(3.0, 7.0)
                logger.info(f"Задержка {delay:.2f} секунд между запросами")
                time.sleep(delay)
            else:
                attempts_without_success += 1  # увеличиваем при неудаче
                
            article_id += 1  # переходим к следующей статье

    logger.info(f"[{code.upper()}] Готово: {len(result)} статей")
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
        articles = get_balanced_articles(code, base_url)

        out_file = OUTPUT_DIR / f"{code}_100.json"  
        with open(out_file, "w", encoding="utf-8") as f:            
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"{code.upper()} — {len(articles)} статей → {out_file.name}\n")
        total += len(articles)

    print(f"\n{'='*60}")
    print(f"ГОТОВО! Собрано {total} статей (5 кодексов)")
    print(f"Папка: ./{OUTPUT_DIR}/")
