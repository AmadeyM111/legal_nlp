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
import urllib3

# ==============================
# Настройка логирования
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'},
                timeout=30,
                verify=False
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
        
        # Extract Title (Robust selectors)
        title_text = "Unknown Title"
        h1 = soup.find("h1", class_="law-element__h1")
        if h1:
            title_text = h1.get_text(strip=True)
        else:
            title_meta = soup.find("meta", property="og:title")
            if title_meta:
                title_text = title_meta["content"]
            elif soup.title:
                title_text = soup.title.get_text(strip=True)

        # Extract Content (Robust selectors)
        content_div = soup.find("div", class_="law-element__body")
        if not content_div:
            content_div = soup.find("div", id="law_text_body")
            
        if content_div:
            # Remove script, style, aside elements
            for script in content_div(["script", "style", "pre", "aside"]):
                script.decompose()
            content = content_div.get_text(separator="\n", strip=True)
        else:
            content = "Content not found"

        return {
            "url": url,
            "title": title_text,
            "content": content
        }


# ==============================
# Основной runner
# ==============================
def scrape_code(strategy: BaseStrategy, start_url: str, limit: int, output_file: str):
    logger.info(f"Начинаем парсинг: {strategy.__class__.__name__}")

    links = strategy.get_article_links(start_url, limit)

    print(f"\n[ОТЛАДКА] Найдено ссылок: {len(links)}")
    if not links:
        print("ОШИБКА: нет ссылок — выходим")
        logger.error("Не удалось получить ссылки на статьи")
        return

    articles = []
    session = requests.Session()
    # Mac User-Agent
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}] → {url}")
        try:
            response = session.get(url, timeout=30, verify=False)
            response.raise_for_status()

            article = strategy.parse_article(url, response.text)
            articles.append(article)

            print(f" Успех: {article['title'][:60]}...")
            time.sleep(random.uniform(0.8, 2.0))

        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")

    print(f"\n[ИТОГ] Успешно спаршено статей: {len(articles)} из {len(links)}")
    
    if len(articles) == 0:
        print("КРИТИЧЕСКАЯ ОШИБКА: ни одна статья не была спаршена!")
        return

    # === SAVE TO FILE ===
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Успешно сохранено {len(articles)} статей → {output_file}")
    print(f"Сохранено → {output_path.resolve()}")


# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Универсальный скрапер кодексов РФ",
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

    # === Конфигурация кодексов ===
    CONFIG = {
        'nk':   ("Налоговый кодекс",    "https://www.zakonrf.info/nk/{}/",     "https://sudact.ru/law/nk-rf/"),
        'tk':   ("Трудовой кодекс",     "https://www.zakonrf.info/tk/{}/",     "https://sudact.ru/law/tk-rf/"),
        'gk':   ("Гражданский кодекс",  "https://www.zakonrf.info/gk/{}/",     "https://sudact.ru/law/gk-rf/"),
        'jk':   ("Жилищный кодекс",     "https://www.zakonrf.info/jk/{}/",     "https://sudact.ru/law/jk-rf/"),
        'koap': ("КоАП РФ",             "https://www.zakonrf.info/koap/{}/",   "https://sudact.ru/law/koap-rf/"),
    }

    if args.code not in CONFIG:
        print(f"Неизвестный кодекс: {args.code}")
        exit(1)

    name, zakonrf_url, sudact_url = CONFIG[args.code]
    base_url = sudact_url if args.site == "sudact" else zakonrf_url

    # Определяем имя файла
    default_filename = f"data/raw/{args.code}_rf_articles.json"
    output_file = args.output or default_filename

    # Выбираем стратегию
    strategy = SudactStrategy() if args.site == "sudact" else ZakonRFStrategy()

    # === ЗАПУСКАЕМ ПАРСИНГ! ===
    print(f"\nЗапускаем парсинг: {name}")
    print(f"Источник: {args.site.upper()}")
    print(f"Статей: {args.articles}")
    print(f"Выходной файл: {output_file}\n")

    scrape_code(strategy, base_url, args.articles, output_file)
