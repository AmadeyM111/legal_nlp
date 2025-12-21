#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скреппер для actual.pravo.gov.ru
Парсит кодексы по структуре HTML с классами Z, T, H

Структура:
- p.Z - заголовок кодекса
- p.T - части/разделы/подразделы (определяются по тексту)
- p.H - главы и статьи (определяются по тексту)
- p (без класса) - пункты статей

Статья = диапазон параграфов от заголовка статьи до следующего заголовка
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pravo_scraper_v2")


class PravoScraperV2:
    """Скреппер для actual.pravo.gov.ru с парсингом структуры кодексов"""
    
    def __init__(self, config_path: str = "scraper_config_v2.json"):
        """Инициализация скреппера"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.codex_index_url = self.config["codex_index_url"]
        self.target_articles = self.config["target_articles_per_code"]
        self.min_content_length = self.config["min_content_length"]
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scraper_config = self.config["scraper"]
        self.codes_config = self.config["codes"]
    
    def get_codex_hash(self, code_key: str) -> Optional[str]:
        """Получает hash кодекса из индекса pravo.gov.ru/codex/"""
        logger.info(f"Получаем hash для кодекса {code_key}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.scraper_config["headless"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            try:
                page.goto(self.codex_index_url, wait_until="networkidle", timeout=self.scraper_config["timeout"])
                
                # Ищем ссылку на кодекс
                code_config = self.codes_config[code_key]
                filter_text = code_config.get("filter_text", "")
                
                # Ищем все ссылки на actual.pravo.gov.ru
                links = page.locator("a[href*='actual.pravo.gov.ru/content/content.html#hash=']")
                count = links.count()
                
                logger.info(f"Найдено {count} ссылок на кодексы")
                
                for i in range(count):
                    link = links.nth(i)
                    href = link.get_attribute("href")
                    text = link.inner_text().strip()
                    
                    if not href:
                        continue
                    
                    # Фильтруем по тексту ссылки
                    if filter_text and filter_text.lower() not in text.lower():
                        # Специальная обработка для КОАП
                        if code_key == "koap" and "коап" not in text.lower():
                            continue
                        elif code_key != "koap":
                            continue
                    
                    # Извлекаем hash
                    match = re.search(r"hash=([0-9a-f]{64})", href)
                    if match:
                        hash_value = match.group(1)
                        logger.info(f"✓ Найден hash для {code_config['name']}: {hash_value[:16]}...")
                        browser.close()
                        return hash_value
                
                logger.warning(f"✗ Hash не найден для {code_key}")
                browser.close()
                return None
                
            except Exception as e:
                logger.error(f"Ошибка при получении hash: {e}")
                browser.close()
                return None
    
    def extract_paragraphs_with_playwright(self, page: Page) -> List[Dict[str, Any]]:
        """Извлекает параграфы напрямую через Playwright селекторы"""
        paragraphs = []
        
        try:
            # Ищем все параграфы
            all_ps = page.locator("p")
            count = all_ps.count()
            logger.info(f"Найдено {count} элементов <p> через Playwright")
            
            for i in range(count):
                try:
                    p_elem = all_ps.nth(i)
                    p_id = p_elem.get_attribute("id") or ""
                    p_class = p_elem.get_attribute("class") or ""
                    text = p_elem.inner_text().strip()
                    
                    if not text:
                        continue
                    
                    # Определяем тип
                    p_type = None
                    if 'Z' in p_class:
                        p_type = 'codex'
                    elif 'T' in p_class:
                        if text.startswith('ЧАСТЬ'):
                            p_type = 'part'
                        elif text.startswith('Раздел'):
                            p_type = 'section'
                        elif text.startswith('Подраздел'):
                            p_type = 'subsection'
                        else:
                            p_type = 't_header'
                    elif 'H' in p_class:
                        if text.startswith('Глава'):
                            p_type = 'chapter'
                        elif text.startswith('Статья'):
                            p_type = 'article_header'
                        else:
                            p_type = 'h_header'
                    else:
                        p_type = 'paragraph'
                    
                    paragraphs.append({
                        'id': p_id,
                        'class': p_class,
                        'type': p_type,
                        'text': text
                    })
                except Exception as e:
                    logger.debug(f"Ошибка при обработке параграфа {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Ошибка при извлечении параграфов через Playwright: {e}")
        
        return paragraphs
    
    def parse_paragraphs(self, html: str) -> List[Dict[str, Any]]:
        """Парсит HTML и извлекает все параграфы с их классами и id"""
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = []
        
        # Находим все параграфы
        for p in soup.find_all('p'):
            p_id = p.get('id', '')
            p_class = p.get('class', [])
            p_class_str = ' '.join(p_class) if isinstance(p_class, list) else str(p_class)
            text = p.get_text(strip=True)
            
            if not text:
                continue
            
            # Определяем тип параграфа
            p_type = None
            if 'Z' in p_class_str:
                p_type = 'codex'
            elif 'T' in p_class_str:
                # Определяем подтип по тексту
                if text.startswith('ЧАСТЬ'):
                    p_type = 'part'
                elif text.startswith('Раздел'):
                    p_type = 'section'
                elif text.startswith('Подраздел'):
                    p_type = 'subsection'
                else:
                    p_type = 't_header'
            elif 'H' in p_class_str:
                if text.startswith('Глава'):
                    p_type = 'chapter'
                elif text.startswith('Статья'):
                    p_type = 'article_header'
                else:
                    p_type = 'h_header'
            else:
                p_type = 'paragraph'
            
            paragraphs.append({
                'id': p_id,
                'class': p_class_str,
                'type': p_type,
                'text': text
            })
        
        return paragraphs
    
    def extract_articles(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Извлекает статьи из списка параграфов
        
        Статья = диапазон от заголовка статьи (p.H с текстом "Статья ...") 
        до следующего заголовка (статья/глава/раздел)
        """
        articles = []
        current_article = None
        
        for i, para in enumerate(paragraphs):
            if para['type'] == 'article_header':
                # Сохраняем предыдущую статью, если есть
                if current_article:
                    articles.append(current_article)
                
                # Начинаем новую статью
                current_article = {
                    'title': para['text'],
                    'header_id': para['id'],
                    'content': [],
                    'paragraphs': []
                }
            
            elif current_article:
                # Проверяем, не закончилась ли статья
                # Статья заканчивается на следующем заголовке (статья, глава, раздел, часть)
                if para['type'] in ['article_header', 'chapter', 'section', 'part', 'subsection', 'codex']:
                    # Сохраняем текущую статью
                    articles.append(current_article)
                    current_article = None
                else:
                    # Добавляем параграф к текущей статье
                    current_article['content'].append(para['text'])
                    current_article['paragraphs'].append({
                        'id': para['id'],
                        'type': para['type'],
                        'text': para['text']
                    })
        
        # Сохраняем последнюю статью, если есть
        if current_article:
            articles.append(current_article)
        
        return articles
    
    def clean_article_content(self, article: Dict[str, Any]) -> str:
        """Очищает и форматирует содержимое статьи"""
        content_parts = []
        
        # Добавляем заголовок
        content_parts.append(article['title'])
        content_parts.append('')
        
        # Добавляем пункты статьи
        for para in article['content']:
            content_parts.append(para)
        
        return '\n'.join(content_parts)
    
    def is_article_valid(self, article: Dict[str, Any]) -> bool:
        """Проверяет актуальность статьи"""
        content = self.clean_article_content(article)
        cleaned = content.strip()
        return len(cleaned) >= self.min_content_length
    
    def stratify_articles(self, articles: List[Dict], target_count: int) -> List[Dict]:
        """Стратифицирует статьи для равномерного распределения"""
        if len(articles) <= target_count:
            return articles
        
        total = len(articles)
        step = total / target_count
        
        stratified = []
        indices = set()
        
        for i in range(target_count):
            idx = int(i * step)
            if idx < total:
                indices.add(idx)
        
        for idx in sorted(indices):
            if idx < len(articles):
                stratified.append(articles[idx])
        
        if len(stratified) < target_count:
            remaining = [a for a in articles if a not in stratified]
            import random
            random.shuffle(remaining)
            stratified.extend(remaining[:target_count - len(stratified)])
        
        return stratified[:target_count]
    
    def scrape_code(self, code_key: str) -> List[Dict[str, Any]]:
        """Собирает статьи для одного кодекса"""
        if code_key not in self.codes_config:
            logger.error(f"Неизвестный кодекс: {code_key}")
            return []
        
        code_config = self.codes_config[code_key]
        logger.info(f"Начинаем сбор статей для {code_config['name']} ({code_key})")
        
        # Получаем hash кодекса
        hash_value = self.get_codex_hash(code_key)
        if not hash_value:
            logger.error(f"Не удалось получить hash для {code_key}")
            return []
        
        # Формируем URL
        codex_url = f"http://actual.pravo.gov.ru/content/content.html#hash={hash_value}&ttl=1"
        logger.info(f"Загружаем кодекс: {codex_url}")
        
        articles = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.scraper_config["headless"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            try:
                # Загружаем страницу кодекса
                logger.info("Загружаем страницу...")
                try:
                    page.goto(codex_url, wait_until="networkidle", timeout=self.scraper_config["timeout"])
                except PlaywrightTimeoutError:
                    logger.warning("Network idle timeout, используем domcontentloaded")
                    page.goto(codex_url, wait_until="domcontentloaded", timeout=self.scraper_config["timeout"])
                
                import time
                time.sleep(3)  # Время для выполнения JavaScript
                
                # Ждем загрузки динамического контента
                logger.info("Ожидаем загрузки контента...")
                try:
                    # Ждем появления контейнера с контентом
                    page.wait_for_selector(".view-col-contaner, iframe, p.Z, p.T, p.H", timeout=10000)
                    logger.info("Контейнер найден, ждем дополнительное время...")
                    time.sleep(3)  # Дополнительное время для загрузки AJAX контента
                except PlaywrightTimeoutError:
                    logger.warning("Таймаут ожидания контейнера, продолжаем...")
                    time.sleep(2)
                
                # Пробуем найти контент в iframe, если есть
                try:
                    iframe_locator = page.locator("iframe").first
                    if iframe_locator.count() > 0:
                        logger.info("Найден iframe, пробуем получить его содержимое...")
                        # Получаем frame через frame_locator
                        frame_locator = page.frame_locator("iframe").first
                        # Пробуем получить HTML из iframe через JavaScript
                        html_iframe = page.evaluate("""
                            () => {
                                const iframe = document.querySelector('iframe');
                                if (iframe && iframe.contentDocument) {
                                    return iframe.contentDocument.documentElement.outerHTML;
                                }
                                return null;
                            }
                        """)
                        if html_iframe:
                            logger.info("Получен HTML из iframe")
                            html = html_iframe
                        else:
                            logger.info("Не удалось получить HTML из iframe, используем основной контент")
                            html = page.content()
                    else:
                        html = page.content()
                except Exception as e:
                    logger.warning(f"Ошибка при работе с iframe: {e}, используем основной контент")
                    html = page.content()
                
                # Отладочный вывод
                logger.debug(f"Размер HTML: {len(html)} символов")
                
                # Парсим параграфы
                logger.info("Парсим структуру кодекса...")
                paragraphs = self.parse_paragraphs(html)
                logger.info(f"Найдено {len(paragraphs)} параграфов")
                
                # Если параграфов нет, пробуем найти их через селекторы Playwright
                if len(paragraphs) == 0:
                    logger.warning("Параграфы не найдены в HTML, пробуем через Playwright селекторы...")
                    paragraphs = self.extract_paragraphs_with_playwright(page)
                    logger.info(f"Найдено {len(paragraphs)} параграфов через Playwright")
                
                # Извлекаем статьи
                logger.info("Извлекаем статьи...")
                all_articles = self.extract_articles(paragraphs)
                logger.info(f"Извлечено {len(all_articles)} статей")
                
                # Фильтруем валидные статьи
                valid_articles = [a for a in all_articles if self.is_article_valid(a)]
                logger.info(f"Валидных статей: {len(valid_articles)}")
                
                # Стратифицируем
                stratified = self.stratify_articles(valid_articles, self.target_articles)
                logger.info(f"После стратификации: {len(stratified)} статей")
                
                # Форматируем для сохранения
                for article in stratified:
                    content = self.clean_article_content(article)
                    articles.append({
                        'url': codex_url,
                        'title': article['title'],
                        'content': content,
                        'code': code_key,
                        'header_id': article.get('header_id', '')
                    })
                
            except Exception as e:
                logger.error(f"Ошибка при парсинге кодекса: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                browser.close()
        
        logger.info(f"✓ Собрано {len(articles)} статей для {code_config['name']}")
        return articles
    
    def save_articles(self, articles: List[Dict], code_key: str):
        """Сохраняет статьи в JSON файл"""
        if not articles:
            logger.warning(f"Попытка сохранить пустой список статей для {code_key}")
            return
        
        output_file = self.output_dir / f"{code_key}_rf_articles.json"
        
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            
            if output_file.exists():
                file_size = output_file.stat().st_size
                logger.info(f"✓ Сохранено {len(articles)} статей в {output_file.resolve()} (размер: {file_size} байт)")
                
                # Проверка
                with open(output_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if len(loaded) == len(articles):
                    logger.info(f"✓ Проверка: файл содержит {len(loaded)} статей")
                else:
                    logger.warning(f"⚠ Несоответствие: файл содержит {len(loaded)} статей, ожидалось {len(articles)}")
            else:
                logger.error(f"✗ Файл не был создан: {output_file}")
                raise FileNotFoundError(f"Файл не был создан: {output_file}")
                
        except Exception as e:
            logger.error(f"✗ Ошибка при сохранении файла {output_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """Точка входа"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Скреппер для actual.pravo.gov.ru",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scraper_config_v2.json",
        help="Путь к файлу конфигурации"
    )
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        choices=["gk", "nk", "tk", "koap", "apk"],
        help="Кодекс для парсинга"
    )
    
    args = parser.parse_args()
    
    scraper = PravoScraperV2(args.config)
    
    try:
        articles = scraper.scrape_code(args.code)
        if articles:
            scraper.save_articles(articles, args.code)
            logger.info(f"✓ Успешно собрано и сохранено {len(articles)} статей для {args.code}")
        else:
            logger.warning(f"✗ Не удалось собрать статьи для {args.code}")
    except Exception as e:
        logger.error(f"✗ Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()

