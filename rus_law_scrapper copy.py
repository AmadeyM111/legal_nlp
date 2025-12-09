#!/usr/bin/env python3
"""
Универсальный скрапер юридических кодексов РФ
Поддерживает: sudact.ru и zakonrf.info
Стратегия: Strategy Pattern — легко добавить новый сайт
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
import time
import random
from abc import ABC, abstractmethod
import argparse
import json
from pathlib import Path

# ==============================
# Настройка логирования
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Отключаем предупреждения SSL только если нужно (лучше не надо, но для sudact.ru бывает)
# requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


# ==============================
# Абстрактная стратегия
# ==============================
class BaseStrategy(ABC):
    @abstractmethod
    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        pass

    @abstractmethod
    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        pass


# ==============================
# Стратегия: sudact.ru
# ==============================
class SudactStrategy(BaseStrategy):
    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        links = []
        try:
            logger.info(f"Загружаем оглавление: {start_url}")
            response = requests.get(
                start_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                timeout=15,
                verify=False  # sudact.ru часто с проблемным SSL
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'statia-' in href.lower() and '/law/' in href:
                    full_url = href if href.startswith('http') else 'https://sudact.ru' + href
                    if full_url not in links:
                        links.append(full_url)
                        if len(links) >= limit:
                            break

            logger.info(f"Найдено {len(links)} ссылок на статьи")
            return links[:limit]

        except Exception as e:
            logger.error(f"Ошибка при получении ссылок: {e}")
            return []

    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Без заголовка"

        content = ""
        container = soup.find('div', class_='law-document-content') or soup.find('div', class_='doc-content')
        if container:
            for p in container.find_all('p'):
                text = p.get_text(strip=True)
                if text:
                    content += text + "\n\n"

        return {
            "url": url,
            "title": title,
            "content": content.strip()
        }


# ==============================
# Стратегия: zakonrf.info
# ==============================
class ZakonRFStrategy(BaseStrategy):
    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        # start_url должен содержать {} для подстановки номера статьи
        # Пример: "https://www.zakonrf.info/nk/{}/"
        links = []
        for i in range(1, limit + 1):
            links.append(start_url.format(i))
        logger.info(f"Сгенерировано {len(links)} ссылок по шаблону")
        return links

    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.select_one('h1.law-element__h1')
        title_text = title.get_text(strip=True) if title else "Без заголовка"

        content = ""
        body = soup.select_one('div.law-element__body')
        if body:
            for p in body.find_all('p'):
                text = p.get_text(strip=True)
                if text:
                    content += text + "\n\n"

        return {
            "url": url,
            "title": title_text,
            "content": content.strip()
        }


# ==============================
# ==============================
# Основной runner
# ==============================
def scrape_code(strategy: BaseStrategy, start_url: str, limit: int, output_file: str):
    logger.info(f"Начинаем парсинг: {strategy.__class__.__name__}")

    links = strategy.get_article_links(start_url, limit)

# Debbug #1 
    print(f"\n[ОТЛАДКА] Найдено ссылок: {len(links)}")
    if not links:
        print("ОШИБКА: нет ссылок — выходим")
        logger.error("Не удалось получить ссылки на статьи")
        return

    articles = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    # Debbug #2: Inside the loop
    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}] → {url}")
        try:
            response = session.get(url, timeout=15, verify=False)
            response.raise_for_status()

            article = strategy.parse_article(url, response.text)

            articles.append(article)

            print(f" Успех: {article['title'][:60]}...")
            # Polite + anti-ban
            time.sleep(random.uniform(0.8, 2.0))

        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")

# ==== DEBUG AFTER SAVE ===
    print(f"\n[ИТОГ] Успешно спаршено статей: {len(articles)} из {len(links)}")
    if len(articles) == 0:
        print("КРИТИЧЕСКАЯ ОШИБКА: ни одна статья не была спаршена!")
        return

    # === SAVE TO FILE ===
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Успешно сохранено {len(articles)} статей → {output_file}")

    print(f"Сохранено → {output_path.resolve()}")
    print(f"\n[ОТЛАДКА] Всего ссылок: {len(links)}")
    print(f"[ОТЛАДКА] Успешно спаршено: {len(articles)}")
    if len(articles) == 0:
        print("[ОШИБКА] Ни одна статья не была спаршена — файл не будет создан!")
        return

# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Универсальный скрапер кодексов РФ",
        epilog="Примеры:\n"
                "  python rus_law_scrapper.py nk                → Налоговый кодекс (150 статей)\n"
                "  python rus_law_scrapper.py tk --articles 50 → ТК РФ, только 50 статей\n"
                "  python rus_law_scrapper.py gk --site sudact → ГК РФ с sudact.ru",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("code", choices=['nk', 'tk', 'gk', 'jk', 'koap'],
                        help="Кодекс для парсинга")
    parser.add_argument("--site", choices=['sudact', 'zakonrf'], default='zakonrf',
                        help="Источник данных (по умолчанию: zakonrf)")
    parser.add_argument("--articles", type=int, default=150,
                        help="Количество статей для парсинга (по умолчанию: 150)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь к выходному файлу")

    args = parser.parse_args()