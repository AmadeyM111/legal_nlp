#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrape Russian codes from official source actual.pravo.gov.ru and build dataset pairs:
{"case": "...", "article": "..."} split into train/test.

Pipeline:
1) Parse http://pravo.gov.ru/codex/ to get document hashes for GK/NK/TK/JK/KOAP [web:464]
2) For each hash:
   - GET redactions -> choose actual redid
   - GET getcontent -> take nodes where unit == "статья" (lvl usually 3) [file:574]
   - GET redtext -> full HTML text (escaped in JSON)
   - Slice each article by np..npe anchors ("p...") [file:574]
3) Convert each article to pair:
   case = title line ("Статья N. ...")
   article = full cleaned text of the article
4) Save JSONL: data/datasets/actual_pravo/<code>/train.jsonl, test.jsonl

Notes:
- This creates "query->law text" style pairs, not real-life case fact patterns.
- Be respectful with request rate.
"""

import json
import random
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote
import urllib3

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# -----------------------------
# Config
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Изменено на DEBUG для детальной отладки
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("actual_pravo_scraper")

# Playwright для динамического скрапинга
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Install with: pip install playwright && playwright install chromium")
# Удаляем его, так как он вызывает ошибки

# Relative paths from script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets" / "actual_pravo"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

CODES = ["gk", "nk", "tk", "jk", "koap"]

# How many pairs per code (after dedupe & filtering)
TARGET_PER_CODE = 300
MIN_CONTENT_LENGTH = 200

TEST_RATIO = 0.1
RANDOM_SEED = 42

SLEEP_RANGE = (0.5, 1.2)

CODEX_INDEX = "http://pravo.gov.ru/codex/"                 # hash discovery [web:464]
# Попробуем оба варианта API
ACTUAL_API = "http://actual.pravo.gov.ru:8000/api/ebpi/"   # content endpoints (observed in DevTools)
ACTUAL_API_ALT = "https://actual.pravo.gov.ru/api/ebpi/"   # альтернативный endpoint (HTTPS)

STOP_LINES = {"Закрыть", "Развернуть", "Свернуть"}


# -----------------------------
# HTTP session
# -----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Origin": "http://actual.pravo.gov.ru",
    "Referer": "http://actual.pravo.gov.ru/",
})
# Отключаем проверку SSL для insecure соединений (как в curl --insecure)
session.verify = False

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException,))
)
def fetch_text(url: str, timeout: int = 60) -> str:
    resp = session.get(url, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.text


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException,))
)
def fetch_json(url: str, timeout: int = 60) -> Dict:
    resp = session.get(url, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.json()


# Удалена дублирующая функция api_redtext - используется версия ниже

def slice_html_by_anchor_range(html: str, start_anchor: str, end_anchor: str) -> str:
    # start_anchor/end_anchor like "p9"
    start_pats = [f'id=\\"{start_anchor}\\"', f'id="{start_anchor}"']
    end_pats   = [f'id=\\"{end_anchor}\\"',   f'id="{end_anchor}"']

    s_idx = -1
    for pat in start_pats:
        s_idx = html.find(pat)
        if s_idx != -1:
            break
    if s_idx == -1:
        return ""

    e_idx = -1
    for pat in end_pats:
        e_idx = html.find(pat, s_idx)
        if e_idx != -1:
            break
    if e_idx == -1:
        return ""

    # Закрывающий </p> может быть как </p> так и <\/p> в экранированном виде
    close_candidates = ["</p>", "<\\/p>"]
    close_idx = -1
    for cc in close_candidates:
        close_idx = html.find(cc, e_idx)
        if close_idx != -1:
            close_idx += len(cc)
            break
    if close_idx == -1:
        close_idx = e_idx

    return html[s_idx:close_idx]


# -----------------------------
# Helpers
# -----------------------------

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def extract_article_number(title: str) -> Optional[str]:
    if not title:
        return None
    m = re.search(r"Статья\s+(\d+(?:\.\d+)*)", title)
    return m.group(1) if m else None


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def train_test_split(items: List[Dict], test_ratio: float, seed: int):
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)
    n_test = max(1, int(len(items) * test_ratio)) if items else 0
    test = items[:n_test]
    train = items[n_test:]
    return train, test


def normalize_code_filter_text(code: str) -> str:
    # Minimal string filters for codex index titles
    if code == "gk":
        return "Гражданский кодекс"
    if code == "nk":
        return "Налоговый кодекс"
    if code == "tk":
        return "Трудовой кодекс"
    if code == "jk":
        return "Жилищный кодекс"
    if code == "koap":
        return "административных правонарушениях"
    return ""


# -----------------------------
# 1) Parse codex index to get hashes
# -----------------------------
def parse_codex_hashes_for_code(code: str) -> List[Dict]:
    html = fetch_text(CODEX_INDEX)
    soup = BeautifulSoup(html, "html.parser")

    anchors = soup.select("a[href*='actual.pravo.gov.ru/content/content.html#hash=']")
    must_contain = normalize_code_filter_text(code)

    out = []
    for a in anchors:
        href = a.get("href")
        if not href:
            continue
        text = clean_text(a.get_text(" ", strip=True))

        # Very light filter by visible text of link
        if must_contain and must_contain not in text and not (code == "koap" and "КоАП" in text):
            continue

        # Ищем hash в URL
        m = re.search(r"hash=([0-9a-f]{64})", href)
        if not m:
            continue

        hash_value = m.group(1)
        
        # Также пытаемся извлечь docid из URL, если есть
        docid_match = re.search(r"docid=(\d+)", href)
        docid = int(docid_match.group(1)) if docid_match else None

        out.append({
            "hash": hash_value, 
            "doc_link_title": text, 
            "doc_url": href,
            "docid": docid  # Сохраняем docid, если найден
        })

    # uniq by hash, keep order
    uniq = []
    seen = set()
    for x in out:
        if x["hash"] not in seen:
            seen.add(x["hash"])
            uniq.append(x)
    
    logger.info(f"[{code.upper()}] Found {len(uniq)} unique documents")
    return uniq


# -----------------------------
# 2) actual.pravo.gov.ru API wrappers
# -----------------------------
def get_redid_from_hash(hash_: str) -> Optional[int]:
    """
    Получает redid из hash через API.
    Пробуем разные варианты endpoints.
    """
    # Вариант 1: попробуем получить redid через getcontent или другой endpoint
    # Но обычно redid получается из redactions endpoint
    pass

def api_redactions(hash_: str, docid: Optional[int] = None, doc_url: Optional[str] = None, ttl: int = 1) -> Dict:
    """
    Endpoint redactions возвращает 404, поэтому пропускаем его.
    Redid будет извлечен из страницы документа напрямую через парсинг.
    """
    logger.debug(f"Skipping redactions API call (endpoint returns 404). Will parse page directly instead.")
    return {"error": "Redactions endpoint not available (404)", "redactions": None}


def get_redid_with_playwright(doc_url: str, hash_value: str, code: str) -> Optional[int]:
    """
    Использует Playwright для получения redid из динамически загруженной страницы.
    Выполняет JavaScript и ждёт загрузки контента.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning(f"[{code.upper()}] Playwright not available, skipping dynamic content extraction")
        return None
    
    try:
        logger.info(f"[{code.upper()}] Using Playwright to extract redid from dynamic content...")
        
        with sync_playwright() as p:
            # Запускаем браузер в headless режиме
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            page = context.new_page()
            
            # Переходим на страницу
            logger.debug(f"[{code.upper()}] Navigating to: {doc_url}")
            page.goto(doc_url, wait_until="networkidle", timeout=30000)
            
            # Ждём загрузки контента (страница загружает textdiv.html через AJAX)
            # Ждём появления элементов или выполнения JavaScript
            try:
                # Ждём, пока загрузится контент (может быть в разных местах)
                page.wait_for_timeout(3000)  # Даём время на выполнение AJAX запросов
                
                # Пробуем найти redid в window объекте или в загруженном контенте
                redid = None
                
                # Метод 1: Ищем в window объекте
                try:
                    redid_js = page.evaluate("""
                        () => {
                            // Ищем redid в различных местах
                            if (window.REQ && window.REQ.redid) return window.REQ.redid;
                            if (window.redid) return window.redid;
                            if (window.rdk) return window.rdk;
                            if (window.DOC && window.DOC.redid) return window.DOC.redid;
                            
                            // Пробуем найти в DOM
                            const scripts = document.querySelectorAll('script');
                            for (let script of scripts) {
                                const text = script.textContent || '';
                                const match = text.match(/(?:redid|rdk)["']?\\s*[:=]\\s*["']?(\\d{4,6})/i);
                                if (match) return parseInt(match[1]);
                            }
                            
                            // Ищем в data-атрибутах
                            const elem = document.querySelector('[data-redid]');
                            if (elem) return parseInt(elem.getAttribute('data-redid'));
                            
                            return null;
                        }
                    """)
                    if redid_js and isinstance(redid_js, (int, str)):
                        try:
                            redid = int(redid_js)
                            if 1000 <= redid <= 999999:
                                logger.info(f"[{code.upper()}] Found redid={redid} via Playwright (window object)")
                                return redid
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
                    logger.debug(f"[{code.upper()}] Error extracting redid from window: {e}")
                
                # Метод 2: Получаем HTML после выполнения JavaScript и парсим
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                
                # Ищем в обновлённом HTML
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string or ""
                    if not script_text:
                        continue
                    
                    # Расширенные паттерны
                    patterns = [
                        r'redid["\']?\s*[:=]\s*["\']?(\d{4,6})',
                        r'rdk["\']?\s*[:=]\s*["\']?(\d{4,6})',
                        r'REQ\.redid\s*=\s*(\d{4,6})',
                        r'REQ\.rdk\s*=\s*(\d{4,6})',
                        r'window\.redid\s*=\s*(\d{4,6})',
                        r'window\.rdk\s*=\s*(\d{4,6})',
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, script_text, re.IGNORECASE)
                        for match in matches:
                            potential_redid = int(match.group(1))
                            if 1000 <= potential_redid <= 999999:
                                redid = potential_redid
                                logger.info(f"[{code.upper()}] Found redid={redid} via Playwright (parsed HTML)")
                                return redid
                
                # Метод 3: Пробуем выполнить JavaScript код, который может получить redid
                try:
                    # Пробуем вызвать функции, которые могут вернуть redid
                    redid_from_js = page.evaluate("""
                        () => {
                            try {
                                // Если есть функция parseRequest, пробуем её использовать
                                if (typeof parseRequest === 'function') {
                                    const hash = window.location.hash.substring(1);
                                    const req = parseRequest(hash);
                                    if (req && req.redid) return req.redid;
                                    if (req && req.rdk) return req.rdk;
                                }
                                
                                // Ищем в глобальных переменных
                                if (typeof REQ !== 'undefined' && REQ.redid) return REQ.redid;
                                if (typeof REQ !== 'undefined' && REQ.rdk) return REQ.rdk;
                                
                                return null;
                            } catch(e) {
                                return null;
                            }
                        }
                    """)
                    if redid_from_js and isinstance(redid_from_js, (int, str)):
                        try:
                            redid = int(redid_from_js)
                            if 1000 <= redid <= 999999:
                                logger.info(f"[{code.upper()}] Found redid={redid} via Playwright (JS execution)")
                                return redid
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    logger.debug(f"[{code.upper()}] Error executing JS to get redid: {e}")
                
                # Метод 4: Ждём загрузки AJAX контента и ищем в нём
                try:
                    # Ждём появления контента (может быть в iframe или загруженном div)
                    page.wait_for_selector(".view-col-contaner, iframe, [data-redid]", timeout=10000)
                    page.wait_for_timeout(2000)  # Дополнительное время для загрузки
                    
                    # Пробуем найти в загруженном контенте
                    redid_from_content = page.evaluate("""
                        () => {
                            // Ищем во всех возможных местах
                            const selectors = [
                                '[data-redid]',
                                '[data-rdk]',
                                '.redid',
                                '#redid'
                            ];
                            
                            for (let sel of selectors) {
                                const elem = document.querySelector(sel);
                                if (elem) {
                                    const val = elem.getAttribute('data-redid') || 
                                               elem.getAttribute('data-rdk') ||
                                               elem.textContent;
                                    const num = parseInt(val);
                                    if (!isNaN(num) && num > 1000 && num < 999999) {
                                        return num;
                                    }
                                }
                            }
                            return null;
                        }
                    """)
                    if redid_from_content and isinstance(redid_from_content, (int, str)):
                        try:
                            redid = int(redid_from_content)
                            if 1000 <= redid <= 999999:
                                logger.info(f"[{code.upper()}] Found redid={redid} via Playwright (content selector)")
                                return redid
                        except (ValueError, TypeError):
                            pass
                except PlaywrightTimeoutError:
                    logger.debug(f"[{code.upper()}] Timeout waiting for content selectors")
                except Exception as e:
                    logger.debug(f"[{code.upper()}] Error waiting for content: {e}")
                
            finally:
                browser.close()
        
        logger.warning(f"[{code.upper()}] Could not extract redid via Playwright")
        return None
        
    except Exception as e:
        logger.error(f"[{code.upper()}] Error using Playwright: {e}", exc_info=True)
        return None


def extract_redid_with_playwright(doc_url: str, timeout: int = 10000) -> Optional[int]:
    """
    Извлекает redid из страницы с помощью Playwright, выполняя JavaScript.
    Дожидается загрузки динамического контента и ищет redid в различных местах.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available, cannot extract redid dynamically")
        return None
    
    try:
        with sync_playwright() as p:
            # Запускаем браузер в headless режиме
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            logger.info(f"Loading page with Playwright: {doc_url}")
            try:
                page.goto(doc_url, wait_until="networkidle", timeout=timeout)
            except PlaywrightTimeoutError:
                logger.warning(f"Network idle timeout, trying with domcontentloaded")
                page.goto(doc_url, wait_until="domcontentloaded", timeout=timeout)
            
            # Ждем выполнения JavaScript и загрузки динамического контента
            logger.debug("Waiting for JavaScript execution...")
            page.wait_for_timeout(3000)  # Увеличено время ожидания
            
            # Ждем загрузки textdiv.html (как видно из лога страницы)
            try:
                page.wait_for_selector('.view-col-contaner', timeout=5000)
                logger.debug("Found .view-col-contaner, waiting additional time for AJAX")
                page.wait_for_timeout(2000)
            except PlaywrightTimeoutError:
                logger.debug("Timeout waiting for .view-col-contaner, continuing anyway")
            
            redid = None
            
            # Метод 1: Ищем redid в window объекте через JavaScript
            try:
                # Пробуем получить redid из различных возможных мест в JavaScript
                js_code = """
                () => {
                    // Ждем выполнения parseRequest
                    if (typeof parseRequest === 'function' && window.location.hash) {
                        try {
                            const hash = window.location.hash.substring(1);
                            const REQ = parseRequest(hash);
                            if (REQ && REQ.redid) return REQ.redid;
                            if (REQ && REQ.rdk) return REQ.rdk;
                            if (REQ && REQ.t) return parseInt(REQ.t);
                        } catch(e) {}
                    }
                    
                    // Ищем в window объекте после выполнения скриптов
                    if (window.REQ) {
                        if (window.REQ.redid) return window.REQ.redid;
                        if (window.REQ.rdk) return window.REQ.rdk;
                        if (window.REQ.t) return parseInt(window.REQ.t);
                    }
                    
                    // Ищем в глобальных переменных
                    if (window.redid) return window.redid;
                    if (window.rdk) return window.rdk;
                    if (typeof redid !== 'undefined') return redid;
                    if (typeof rdk !== 'undefined') return rdk;
                    
                    // Парсим из location.hash вручную (если parseRequest не работает)
                    if (window.location.hash) {
                        const hash = window.location.hash.substring(1);
                        // Пробуем URLSearchParams
                        try {
                            const params = new URLSearchParams(hash);
                            if (params.get('redid')) return parseInt(params.get('redid'));
                            if (params.get('rdk')) return parseInt(params.get('rdk'));
                            if (params.get('t')) return parseInt(params.get('t'));
                        } catch(e) {}
                        
                        // Парсим вручную через regex
                        const redidMatch = hash.match(/[&?]redid=(\d+)/i);
                        if (redidMatch) return parseInt(redidMatch[1]);
                        const rdkMatch = hash.match(/[&?]rdk=(\d+)/i);
                        if (rdkMatch) return parseInt(rdkMatch[1]);
                        const tMatch = hash.match(/[&?]t=(\d+)/i);
                        if (tMatch) return parseInt(tMatch[1]);
                    }
                    
                    return null;
                }
                """
                # Ждем выполнения скриптов на странице
                page.wait_for_timeout(1000)
                result = page.evaluate(js_code)
                logger.debug(f"JavaScript evaluation result: {result} (type: {type(result)})")
                if result:
                    try:
                        redid = int(result) if not isinstance(result, int) else result
                        if 1000 <= redid <= 999999:
                            logger.info(f"Found redid={redid} via JavaScript evaluation")
                            return redid
                        else:
                            logger.debug(f"Redid {redid} is outside valid range (1000-999999)")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Failed to convert result to int: {result}, error: {e}")
            except Exception as e:
                logger.warning(f"JavaScript evaluation failed: {e}", exc_info=True)
            
            # Метод 2: Ищем в DOM элементах (data-атрибуты)
            if not redid:
                try:
                    data_redid = page.get_attribute('[data-redid]', 'data-redid')
                    if data_redid:
                        redid = int(data_redid)
                        logger.info(f"Found redid={redid} in data-redid attribute")
                except Exception as e:
                    logger.debug(f"data-redid search failed: {e}")
            
            # Метод 3: Ищем в тексте страницы после загрузки
            if not redid:
                try:
                    content = page.content()
                    # Ищем паттерны redid в загруженном HTML
                    patterns = [
                        r'redid["\']?\s*[:=]\s*["\']?(\d{4,6})',
                        r'rdk["\']?\s*[:=]\s*["\']?(\d{4,6})',
                        r'["\']redid["\']\s*:\s*(\d{4,6})',
                        r'REQ\.redid\s*=\s*(\d{4,6})',
                        r'REQ\.rdk\s*=\s*(\d{4,6})',
                    ]
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            potential_redid = int(match.group(1))
                            if 1000 <= potential_redid <= 999999:
                                redid = potential_redid
                                logger.info(f"Found redid={redid} in page content using pattern: {pattern}")
                                break
                        if redid:
                            break
            except Exception as e:
                logger.debug(f"Content search failed: {e}")
            
            # Если redid все еще не найден, выводим отладочную информацию
            if not redid:
                try:
                    current_url = page.url
                    current_hash = page.evaluate("() => window.location.hash")
                    logger.debug(f"Current URL: {current_url}")
                    logger.debug(f"Current hash: {current_hash}")
                    
                    # Проверяем, есть ли REQ объект
                    req_obj = page.evaluate("() => typeof window.REQ !== 'undefined' ? window.REQ : null")
                    logger.debug(f"REQ object: {req_obj}")
                except Exception as e:
                    logger.debug(f"Failed to get debug info: {e}")
            
            # Метод 4: Перехватываем сетевые запросы для поиска redid в API ответах
            if not redid:
                try:
                    # Перехватываем ответы от API
                    api_responses = []
                    def handle_response(response):
                        try:
                            if 'api/ebpi' in response.url:
                                api_responses.append({
                                    'url': response.url,
                                    'status': response.status
                                })
                                # Пробуем получить JSON из ответа
                                try:
                                    json_data = response.json()
                                    if isinstance(json_data, dict):
                                        # Ищем redid в ответе
                                        if 'redid' in json_data:
                                            api_responses[-1]['redid'] = json_data['redid']
                                        if 'rdk' in json_data:
                                            api_responses[-1]['rdk'] = json_data['rdk']
                                        if 'docid' in json_data and json_data.get('docid', 0) > 0:
                                            api_responses[-1]['docid'] = json_data['docid']
                                except:
                                    pass
                        except:
                            pass
                    
                    page.on("response", handle_response)
                    
                    # Ждем загрузки textdiv.html и выполнения AJAX запросов
                    try:
                        page.wait_for_selector('.view-col-contaner', timeout=5000)
                    except:
                        pass
                    page.wait_for_timeout(3000)  # Дополнительное время для AJAX
                    
                    # Проверяем перехваченные ответы
                    for resp in api_responses:
                        if 'redid' in resp:
                            redid = int(resp['redid'])
                            logger.info(f"Found redid={redid} in API response: {resp['url']}")
                            break
                        if 'rdk' in resp and isinstance(resp['rdk'], (int, str)):
                            try:
                                redid = int(resp['rdk'])
                                if 1000 <= redid <= 999999:
                                    logger.info(f"Found redid={redid} as rdk in API response: {resp['url']}")
                                    break
                            except:
                                pass
                        if 'docid' in resp and resp['docid'] > 1000:
                            redid = int(resp['docid'])
                            logger.info(f"Found redid={redid} as docid in API response: {resp['url']}")
                            break
                    
                    # Также ищем в загруженном контенте
                    if not redid:
                        content = page.content()
                        # Ищем в JSON данных, которые могли загрузиться
                        json_patterns = [
                            r'["\']redid["\']\s*:\s*(\d{4,6})',
                            r'["\']rdk["\']\s*:\s*(\d{4,6})',
                            r'redid["\']?\s*[:=]\s*["\']?(\d{4,6})',
                        ]
                        for pattern in json_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                potential_redid = int(match.group(1))
                                if 1000 <= potential_redid <= 999999:
                                    redid = potential_redid
                                    logger.info(f"Found redid={redid} in page content using pattern: {pattern}")
                                    break
                            if redid:
                                break
                except Exception as e:
                    logger.debug(f"AJAX content wait failed: {e}")
            
            browser.close()
            
            return redid if redid and 1000 <= redid <= 999999 else None
            
    except PlaywrightTimeoutError:
        logger.warning(f"Playwright timeout while loading {doc_url}")
        return None
    except Exception as e:
        logger.error(f"Playwright error: {e}", exc_info=True)
        return None


def pick_actual_redid(redactions_json: Dict) -> Optional[int]:
    """
    Извлекает actual redid из ответа API.
    Также проверяет docid, если redactions пустой.
    """
    # Проверяем наличие ошибки
    if redactions_json.get("error"):
        error_msg = redactions_json.get("error", "")
        logger.warning(f"API returned error: {error_msg}")
        
        # Если есть docid, попробуем использовать его напрямую
        docid = redactions_json.get("docid")
        if docid and docid > 0:
            logger.info(f"Using docid={docid} as redid (fallback)")
            return int(docid)
        
        return None
    
    reds = redactions_json.get("redactions") or []
    if not reds:
        # Если redactions пустой, но есть docid, используем его
        docid = redactions_json.get("docid")
        if docid and docid > 0:
            logger.info(f"Using docid={docid} as redid (no redactions list)")
            return int(docid)
        return None
    
    for r in reds:
        if r.get("actual") is True and r.get("hascontent") is True:
            return int(r["redid"])
    for r in reds:
        if r.get("hascontent") is True:
            return int(r["redid"])
    return None


def api_getcontent(redid: int) -> Dict:
    """
    Получает структуру документа по redid.
    Формат: rdk={redid} (redid как число)
    """
    url = f"{ACTUAL_API}getcontent/?bpa=ebpi&rdk={redid}"
    try:
    return fetch_json(url)
    except Exception as e:
        logger.error(f"Failed to fetch getcontent for redid={redid}: {e}")
        # Пробуем альтернативный endpoint
        try:
            url_alt = f"{ACTUAL_API_ALT}getcontent/?bpa=ebpi&rdk={redid}"
            return fetch_json(url_alt)
        except Exception as e2:
            logger.error(f"Alternative endpoint also failed: {e2}")
            return {}


def api_redtext(redid: int, ttl: int = 1) -> str:
    """
    Получает redtext по redid (число, не hash!).
    Формат как в DevTools: t=483281 (redid как число)
    """
    url = f"{ACTUAL_API}redtext?bpa=ebpi&t={redid}&ttl={ttl}"
    try:
    data = fetch_json(url)
    return data.get("redtext") or ""
    except Exception as e:
        logger.error(f"Failed to fetch redtext for redid={redid}: {e}")
        # Пробуем альтернативный endpoint
        try:
            url_alt = f"{ACTUAL_API_ALT}redtext?bpa=ebpi&t={redid}&ttl={ttl}"
            data = fetch_json(url_alt)
            return data.get("redtext") or ""
        except Exception as e2:
            logger.error(f"Alternative endpoint also failed: {e2}")
            return ""


# -----------------------------
# 3) getcontent -> article nodes (no heuristics)
# -----------------------------
def iter_article_nodes_from_getcontent(getcontent_json: Dict) -> List[Dict]:
    """
    Strict:
    - expects getcontent_json["data"] list
    - article nodes: unit == "статья"
    """
    data = getcontent_json.get("data", [])
    if not isinstance(data, list):
        logger.warning(f"getcontent data is not a list: {type(data)}. Keys: {list(getcontent_json.keys()) if isinstance(getcontent_json, dict) else 'not a dict'}")
        return []
    
    nodes = [it for it in data if isinstance(it, dict) and it.get("unit") == "статья"]
    
    if not nodes:
        # Попробуем найти статьи с другими вариантами unit
        all_units = set(it.get("unit") for it in data if isinstance(it, dict))
        logger.debug(f"No articles found with unit='статья'. Available units: {all_units}")
        # Попробуем альтернативные варианты
        nodes = [it for it in data if isinstance(it, dict) and (
            it.get("unit", "").lower() == "статья" or 
            "статья" in str(it.get("unit", "")).lower() or
            it.get("lvl") == 3  # Статьи обычно на уровне 3
        )]
        logger.debug(f"Found {len(nodes)} nodes with alternative criteria")
    
    return nodes


# -----------------------------
# 4) redtext slicing by anchor range
# -----------------------------
def slice_html_by_anchor_range(redtext_html_escaped: str, start_anchor: str, end_anchor: str) -> str:
    """
    redtext is a big escaped HTML string. We slice by finding id=\"pNN\" fragments.
    start_anchor/end_anchor like "p9", "p5709".
    """
    start_pat = re.compile(rf'id=["\']{re.escape(start_anchor)}["\']')
    end_pat = re.compile(rf'id=["\']{re.escape(end_anchor)}["\']')

    start_match = start_pat.search(redtext_html_escaped)
    if not start_match:
        return ""
    s_idx = start_match.start()

    end_match = end_pat.search(redtext_html_escaped, s_idx)
    if not end_match:
        return ""
    e_idx = end_match.start()
    
    close = redtext_html_escaped.find("</p>", e_idx)
    close_idx = close + len("</p>") if close != -1 else e_idx

    return redtext_html_escaped[s_idx:close_idx]


def unescape_minimal(s: str) -> str:
    # Enough to feed BeautifulSoup
    return (
        s.replace('\\"', '"')
         .replace("\\r\\n", "\n")
         .replace("\\n", "\n")
         .replace("\\t", "\t")
    )


def html_fragment_to_text(fragment_html_escaped: str) -> str:
    html = unescape_minimal(fragment_html_escaped)
    soup = BeautifulSoup(html, "html.parser")

    parts = []
    for line in soup.get_text("\n", strip=True).splitlines():
        t = clean_text(line)
        if not t or t in STOP_LINES:
            continue
        parts.append(t)
    return clean_text("\n".join(parts))


# -----------------------------
# 5) Build article record -> pair
# -----------------------------
def build_article_record(code: str, doc_hash: str, redid: int, doc_title: str, node: Dict, redtext_html: str) -> Optional[Dict]:
    """
    node example:
    {"id":"...","np":"p9","npe":"p5709","caption":"Статья 1. ...","unit":"статья","lvl":3}
    """
    np_ = node.get("np")
    npe_ = node.get("npe")
    if not np_ or not npe_:
        return None

    fragment = slice_html_by_anchor_range(redtext_html, np_, npe_)
    if not fragment:
        return None

    text = html_fragment_to_text(fragment)
    if len(text) < MIN_CONTENT_LENGTH:
        return None

    title = text.splitlines()[0] if text else ""
    article_number = extract_article_number(title) or extract_article_number(node.get("caption", ""))

    return {
        "code": code,
        "doc_hash": doc_hash,
        "redid": redid,
        "doc_title": doc_title,
        "article_number": article_number,
        "title": title,
        "content": text,
        "source": "actual.pravo.gov.ru",
        "fetch_date": time.strftime("%Y-%m-%d"),
    }


def article_record_to_pair(rec: Dict) -> Dict:
    # Minimal pair format requested
    return {"case": rec.get("title", "").strip(), "article": rec.get("content", "").strip()}


def dedupe_pairs(pairs: List[Dict]) -> List[Dict]:
    """
    Dedupe by (case, article). Keep the longest article if same case repeats.
    """
    best = {}
    for p in pairs:
        key = p.get("case", ""),  # case only; "article" may differ between parts/editions
        # If same case repeats, keep longer article
        if key not in best or len(p.get("article", "")) > len(best[key].get("article", "")):
            best[key] = p
    return list(best.values())


# -----------------------------
# 6) Scrape per code
# -----------------------------
def scrape_actual_code_pairs(code: str, target: int) -> List[Dict]:
    docs = parse_codex_hashes_for_code(code)
    logger.info(f"[{code.upper()}] hashes found: {len(docs)}")
    
    if not docs:
        logger.warning(f"[{code.upper()}] No documents found! Check parse_codex_hashes_for_code()")
        return []

    article_records: List[Dict] = []

    for d in docs:
        doc_hash = d["hash"]
        doc_title = d.get("doc_link_title", "")
        docid = d.get("docid")  # Пробуем использовать docid, если есть

        try:
            logger.debug(f"[{code.upper()}] Processing hash={doc_hash[:16]}... | title={doc_title[:50]} | docid={docid}")
            
            doc_url = d.get("doc_url")
            redid = None
            
            logger.debug(f"[{code.upper()}] doc_url={doc_url}, docid={docid}, PLAYWRIGHT_AVAILABLE={PLAYWRIGHT_AVAILABLE}")
            
            # Стратегия 1: Сначала пробуем Playwright для динамического контента (приоритет)
            if doc_url:
                if PLAYWRIGHT_AVAILABLE:
                    logger.info(f"[{code.upper()}] Using Playwright to extract redid from dynamic page")
                    redid = extract_redid_with_playwright(doc_url)
                    if redid:
                        logger.info(f"[{code.upper()}] Successfully extracted redid={redid} with Playwright")
                    else:
                        logger.warning(f"[{code.upper()}] Playwright did not find redid")
                else:
                    logger.warning(f"[{code.upper()}] Playwright not available, skipping dynamic extraction")
            else:
                logger.warning(f"[{code.upper()}] No doc_url available, cannot extract redid")
            
            # Стратегия 2: Если есть docid и Playwright не помог, пробуем использовать docid напрямую
            if not redid and docid:
                logger.info(f"[{code.upper()}] Trying to use docid={docid} directly as redid")
                redid = docid
            
            # Если Playwright не помог или недоступен, пробуем статический парсинг
            if doc_url and not redid:
                logger.info(f"[{code.upper()}] Parsing document page statically to extract redid from HTML")
                try:
                    html = fetch_text(doc_url)
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Метод 1: Ищем в data-атрибутах
                    for elem in soup.find_all(attrs={"data-redid": True}):
                        redid = int(elem.get("data-redid"))
                        logger.info(f"[{code.upper()}] Found redid={redid} in data-redid attribute")
                        break
                    
                    # Метод 2: Ищем в мета-тегах
            if not redid:
                        for meta in soup.find_all("meta"):
                            content = meta.get("content", "")
                            name = meta.get("name", "").lower()
                            if "redid" in name or "rdk" in name:
                                match = re.search(r'(\d+)', content)
                                if match:
                                    redid = int(match.group(1))
                                    logger.info(f"[{code.upper()}] Found redid={redid} in meta tag: {name}")
                                    break
                    
                    # Метод 3: Ищем в JavaScript коде (более детальный поиск)
                    if not redid:
                        scripts = soup.find_all("script")
                        for script in scripts:
                            script_text = script.string or ""
                            if not script_text:
                                continue
                            
                            # Расширенные паттерны для поиска redid
                            patterns = [
                                r'redid["\']?\s*[:=]\s*["\']?(\d+)',
                                r'rdk["\']?\s*[:=]\s*["\']?(\d+)',
                                r'redid\s*=\s*(\d+)',
                                r'rdk\s*=\s*(\d+)',
                                r'["\']rdk["\']\s*:\s*(\d+)',
                                r'redid["\']?\s*:\s*(\d+)',
                                r'redid["\']?\s*=\s*["\']?(\d+)',
                                r'rdk["\']?\s*=\s*["\']?(\d+)',
                                r'redid["\']?\s*:\s*["\']?(\d+)',
                                r'["\']redid["\']\s*:\s*(\d+)',
                                r'redid["\']?\s*["\']?\s*:\s*(\d+)',
                                # Ищем числа после ключевых слов
                                r'(?:redid|rdk)[\s:="\']+(\d{4,})',  # redid должен быть достаточно большим числом
                            ]
                            
                            for pattern in patterns:
                                matches = re.finditer(pattern, script_text, re.IGNORECASE | re.MULTILINE)
                                for match in matches:
                                    potential_redid = int(match.group(1))
                                    # redid обычно больше 1000
                                    if potential_redid > 1000:
                                        redid = potential_redid
                                        logger.info(f"[{code.upper()}] Found redid={redid} in JavaScript using pattern: {pattern}")
                                        break
                                if redid:
                                    break
                            if redid:
                                break
                    
                    # Метод 4: Ищем в URL параметрах самой страницы
                    if not redid:
                        url_match = re.search(r'[?&#](?:redid|rdk|docid|t)=(\d+)', doc_url)
                        if url_match:
                            redid = int(url_match.group(1))
                            logger.info(f"[{code.upper()}] Found redid={redid} in URL parameters")
                    
                    # Метод 4b: Ищем в hash фрагменте URL (после #), но НЕ в самом hash значении
                    if not redid:
                        hash_fragment = doc_url.split('#')[-1] if '#' in doc_url else ""
                        # Исключаем сам hash (64 hex символа) и ищем числа в других параметрах
                        # Убираем hash=... из фрагмента
                        hash_fragment_clean = re.sub(r'hash=[0-9a-f]{64}', '', hash_fragment, flags=re.IGNORECASE)
                        # Ищем числа в оставшихся параметрах (rdk, redid, docid, t)
                        param_match = re.search(r'(?:rdk|redid|docid|t)=(\d{4,6})', hash_fragment_clean, re.IGNORECASE)
                        if param_match:
                            num = int(param_match.group(1))
                            if 1000 <= num <= 999999:  # Разумный диапазон для redid
                                redid = num
                                logger.info(f"[{code.upper()}] Found potential redid={redid} in URL parameters (not hash)")
                                break
                    
                    # Метод 5: Ищем в тексте страницы (может быть в комментариях или скрытых полях)
                    if not redid:
                        page_text = soup.get_text()
                        # Ищем числа, которые могут быть redid (обычно 4-6 цифр)
                        potential_ids = re.findall(r'\b(\d{4,6})\b', page_text)
                        for pid in potential_ids:
                            pid_int = int(pid)
                            # redid обычно в диапазоне 1000-999999
                            if 1000 <= pid_int <= 999999:
                                # Проверяем контекст вокруг числа
                                idx = page_text.find(pid)
                                context = page_text[max(0, idx-20):idx+len(pid)+20].lower()
                                if 'redid' in context or 'rdk' in context or 'doc' in context:
                                    redid = pid_int
                                    logger.info(f"[{code.upper()}] Found potential redid={redid} in page text context")
                                    break
                    
                    if not redid:
                        logger.warning(f"[{code.upper()}] Could not extract redid from page. Attempting alternative methods...")
                        
                        # Сохраняем HTML для отладки (первые 10000 символов)
                        debug_html = html[:10000]
                        logger.debug(f"First 10000 chars of HTML:\n{debug_html}")
                        
                        # Пробуем найти все числа в JavaScript и проверить их как потенциальные redid
                        all_scripts_text = "\n".join([s.string or "" for s in soup.find_all("script")])
                        numbers = re.findall(r'\b(\d{4,6})\b', all_scripts_text)
                        logger.debug(f"Found {len(numbers)} potential numbers in scripts: {set(numbers[:20])}")
                        
                        # Пробуем найти в window/global объектах
                        window_patterns = [
                            r'window\.redid\s*=\s*(\d+)',
                            r'window\.rdk\s*=\s*(\d+)',
                            r'var\s+redid\s*=\s*(\d+)',
                            r'var\s+rdk\s*=\s*(\d+)',
                            r'let\s+redid\s*=\s*(\d+)',
                            r'const\s+redid\s*=\s*(\d+)',
                        ]
                        for pattern in window_patterns:
                            match = re.search(pattern, all_scripts_text, re.IGNORECASE)
                            if match:
                                redid = int(match.group(1))
                                logger.info(f"[{code.upper()}] Found redid={redid} using window pattern: {pattern}")
                                break
                        
                except Exception as e:
                    logger.warning(f"[{code.upper()}] Failed to parse document page: {e}", exc_info=True)
            
            # Стратегия 4: Используем Playwright для получения redid из динамически загруженного контента
            if not redid and doc_url:
                logger.info(f"[{code.upper()}] Attempting to extract redid using Playwright (dynamic content)...")
                playwright_redid = get_redid_with_playwright(doc_url, doc_hash, code)
                if playwright_redid:
                    redid = playwright_redid
            
            # Стратегия 5: Попробуем использовать hash для получения redid через другие API endpoints
            if not redid:
                logger.warning(f"[{code.upper()}] Could not extract redid from page. Trying alternative API approaches...")
                
                # Попробуем использовать hash для получения информации о документе
                # Может быть есть endpoint, который возвращает redid по hash
                try:
                    # Пробуем разные варианты API запросов с hash
                    test_endpoints = [
                        f"{ACTUAL_API}getcontent/?bpa=ebpi&hash={doc_hash}",
                        f"{ACTUAL_API}getcontent/?bpa=ebpi&dochash={doc_hash}",
                        f"{ACTUAL_API}docinfo/?bpa=ebpi&hash={doc_hash}",
                        f"{ACTUAL_API}docinfo/?bpa=ebpi&dochash={doc_hash}",
                    ]
                    for test_url in test_endpoints:
                        try:
                            test_result = fetch_json(test_url)
                            # Ищем redid в ответе
                            if isinstance(test_result, dict):
                                # Проверяем разные возможные поля
                                potential_redid = (test_result.get("redid") or 
                                                 test_result.get("rdk") or 
                                                 test_result.get("docid"))
                                if potential_redid and isinstance(potential_redid, (int, str)):
                                    try:
                                        redid = int(potential_redid)
                                        if 1000 <= redid <= 999999:
                                            logger.info(f"[{code.upper()}] Found redid={redid} in API response from {test_url}")
                                            break
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Если есть data и это список, возможно redid в другом месте
                                if test_result.get("data") and isinstance(test_result.get("data"), list):
                                    logger.debug(f"[{code.upper()}] Got data from {test_url}, but no redid found")
                        except Exception as e:
                            logger.debug(f"[{code.upper()}] Test endpoint {test_url} failed: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"[{code.upper()}] Alternative API approach failed: {e}")
            
            # Если все еще нет redid, сохраняем HTML для анализа
            if not redid:
                logger.error(f"[{code.upper()}] Could not extract redid for hash={doc_hash[:16]}... | doc_url={doc_url}")
                
                # Сохраняем полный HTML для анализа
                try:
                    debug_dir = RAW_DIR / "debug_html"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_file = debug_dir / f"{code}_{doc_hash[:16]}.html"
                    page_html = html if 'html' in locals() else fetch_text(doc_url)
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(page_html)
                    logger.info(f"[{code.upper()}] Saved HTML for debugging: {debug_file}")
                    
                    # Также сохраняем все найденные числа из JavaScript для анализа
                    if 'soup' in locals():
                        all_scripts = "\n".join([s.string or "" for s in soup.find_all("script")])
                        all_numbers = re.findall(r'\b(\d{3,7})\b', all_scripts)
                        numbers_file = debug_dir / f"{code}_{doc_hash[:16]}_numbers.txt"
                        with open(numbers_file, 'w', encoding='utf-8') as f:
                            f.write(f"Found {len(all_numbers)} numbers in scripts:\n")
                            f.write("\n".join(set(all_numbers)))
                        logger.info(f"[{code.upper()}] Saved found numbers: {numbers_file}")
                except Exception as e:
                    logger.debug(f"Failed to save debug files: {e}")
                
                logger.error(f"[{code.upper()}] Skipping this document - cannot proceed without redid")
                continue

            logger.debug(f"[{code.upper()}] Got redid={redid}, fetching getcontent...")
            gc = api_getcontent(redid)
            logger.debug(f"[{code.upper()}] Getcontent response keys: {list(gc.keys()) if isinstance(gc, dict) else 'not a dict'}")
            
            # Проверяем, правильный ли redid
            if gc.get("error"):
                logger.warning(f"[{code.upper()}] getcontent returned error: {gc.get('error')} for redid={redid}")
                # Если redid неправильный, это может быть число из hash
                if str(redid) in doc_hash[:8]:  # Проверяем, не является ли redid частью hash
                    logger.error(f"[{code.upper()}] Redid {redid} appears to be extracted from hash incorrectly. Skipping.")
                    continue
            
            # Проверяем, есть ли data
            if gc.get("data") is None:
                logger.warning(f"[{code.upper()}] getcontent returned data=None for redid={redid}. Redid might be incorrect.")
                # Если redid был найден из hash фрагмента, это скорее всего неправильно
                if str(redid) in doc_hash:
                    logger.error(f"[{code.upper()}] Redid {redid} is part of hash, not a real redid. Skipping.")
                    continue
            
            nodes = iter_article_nodes_from_getcontent(gc)
            logger.info(f"[{code.upper()}] {doc_title[:60]} | redid={redid} | articles={len(nodes)}")
            
            if not nodes:
                logger.warning(f"[{code.upper()}] No article nodes found in getcontent for redid={redid}")
                # Если data пустая, это может означать неправильный redid
                if gc.get("data") is None or (isinstance(gc.get("data"), list) and len(gc.get("data", [])) == 0):
                    logger.warning(f"[{code.upper()}] Empty data in getcontent response. Redid {redid} might be incorrect.")
                continue

            logger.debug(f"[{code.upper()}] Fetching redtext for redid={redid}...")
            rt = api_redtext(redid, ttl=1)
            if not rt:
                logger.warning(f"[{code.upper()}] Empty redtext for redid={redid}")
                continue
            
            logger.debug(f"[{code.upper()}] Redtext length: {len(rt)} chars")

            for node in nodes:
                if len(article_records) >= target * 3:  # safety buffer
                    break
                rec = build_article_record(code, doc_hash, redid, doc_title, node, rt)
                if rec:
                    article_records.append(rec)
                else:
                    logger.debug(f"[{code.upper()}] build_article_record returned None for node np={node.get('np')} npe={node.get('npe')}")

            logger.info(f"[{code.upper()}] Collected {len(article_records)} article records so far")
            time.sleep(random.uniform(*SLEEP_RANGE))

        except Exception as e:
            logger.error(f"[{code.upper()}] error for hash={doc_hash[:16]}...: {e}", exc_info=True)
            time.sleep(random.uniform(*SLEEP_RANGE))

    logger.info(f"[{code.upper()}] Total article records collected: {len(article_records)}")
    return article_records

# -----------------------------
# Main
# -----------------------------
def main():
    stats = {}

    for code in CODES:
        logger.info("=" * 70)
        logger.info(f"Processing code: {code.upper()}")
        
        # Получаем article_records
        article_records = scrape_actual_code_pairs(code, TARGET_PER_CODE)
        logger.info(f"[{code.upper()}] Collected {len(article_records)} article records")
        
        if not article_records:
            logger.warning(f"[{code.upper()}] No article records collected! Skipping...")
            stats[code] = 0
            continue
        
        # Преобразуем в пары
        pairs = [article_record_to_pair(rec) for rec in article_records]
        logger.info(f"[{code.upper()}] Converted to {len(pairs)} pairs")
        
        # Дедупликация
        pairs = dedupe_pairs(pairs)
        logger.info(f"[{code.upper()}] After deduplication: {len(pairs)} pairs")
        
        stats[code] = len(pairs)
        
        if not pairs:
            logger.warning(f"[{code.upper()}] No pairs after processing! Skipping save...")
            continue

        # Split
        train, test = train_test_split(pairs, TEST_RATIO, RANDOM_SEED)

        # Save per-code
        out_dir = DATASETS_DIR / code
        out_dir.mkdir(parents=True, exist_ok=True)

        save_jsonl(out_dir / "train.jsonl", train)
        save_jsonl(out_dir / "test.jsonl", test)

        # Also save a raw JSON (debug)
        save_json(RAW_DIR / f"actual_{code}_{len(pairs)}.json", pairs)

        logger.info(f"[{code.upper()}] pairs={len(pairs)} | train={len(train)} test={len(test)} | saved -> {out_dir}")

    # Save combined dataset
    all_train = []
    all_test = []
    for code in CODES:
        out_dir = DATASETS_DIR / code
        if (out_dir / "train.jsonl").exists():
            all_train.extend([json.loads(x) for x in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
        if (out_dir / "test.jsonl").exists():
            all_test.extend([json.loads(x) for x in (out_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])

    save_jsonl(DATASETS_DIR / "train_all.jsonl", all_train)
    save_jsonl(DATASETS_DIR / "test_all.jsonl", all_test)
    logger.info(f"Done. Stats: {stats} | all_train={len(all_train)} all_test={len(all_test)}")


if __name__ == "__main__":
    main()
