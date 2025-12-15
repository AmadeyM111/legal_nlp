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
logger = logging.getLogger("actual_pravo_scraper")

# Этот блок кода находится вне функции и не использует определенные переменные
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
TARGET_PER_CODE = 150
MIN_CONTENT_LENGTH = 200

TEST_RATIO = 0.1
RANDOM_SEED = 42

SLEEP_RANGE = (0.5, 1.2)

CODEX_INDEX = "http://pravo.gov.ru/codex/"                 # hash discovery [web:464]
ACTUAL_API = "http://actual.pravo.gov.ru:8000/api/ebpi/"   # content endpoints (observed in DevTools)

STOP_LINES = {"Закрыть", "Развернуть", "Свернуть"}


# -----------------------------
# HTTP session
# -----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PythonScraper/1.0",
    "Accept": "*/*",
    "Accept-Language": "ru-RU,ru;q=0.9",
})

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException,))
)
def fetch_text(url: str, timeout: int = 60) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException,))
)
def fetch_json(url: str, timeout: int = 60) -> Dict:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_redtext(redid: int, ttl: int = 1) -> str:
    t = json.dumps({"rdk": redid, "ttl": ttl})
    url = f"{ACTUAL_API}redtext?bpa=ebpi&t={quote(t)}"
    data = fetch_json(url)
    return data.get("redtext", "")

    # Иногда это JSON с ключом "redtext", иногда может быть сразу текст
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        data = resp.json()
        return data.get("redtext") or ""
    return resp.text or ""

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
    # Удаляем HTML id атрибуты в начале строки
    s = re.sub(r'^id="[^"]*"\s*', '', s)
    # Удаляем символ переноса строки
    s = re.sub(r"\r\n?", "\n")
    # Удаляем символ ">" в начале строки, который может оставаться после обработки HTML
    s = re.sub(r'^>\s*', '', s)
    # Заменяем неразрывные пробелы и другие специальные символы
    s = s.replace("\xa0", " ")
    # Заменяем множественные пробелы и табуляции на одиночный пробел
    s = re.sub(r"[ \t]+", " ", s)
    # Заменяем множественные переносы строк на один перенос
    s = re.sub(r"\n+", " ", s)
    # Удаляем лишние пробелы, которые могли образоваться после замены переносов строк
    s = re.sub(r" +", " ", s)
    return s.strip()

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

        m = re.search(r"hash=([0-9a-f]{64})", href)
        if not m:
            continue

        out.append({"hash": m.group(1), "doc_link_title": text, "doc_url": href})

    # uniq by hash, keep order
    uniq = []
    seen = set()
    for x in out:
        if x["hash"] not in seen:
            seen.add(x["hash"])
            uniq.append(x)
    return uniq


# -----------------------------
# 2) actual.pravo.gov.ru API wrappers
# -----------------------------
def api_redactions(hash_: str, ttl: int = 1) -> Dict:
    # t is JSON serialized into query param, as observed in DevTools
    t = json.dumps({"hash": hash_, "ttl": str(ttl)}, ensure_ascii=False)
    url = f"{ACTUAL_API}redactions/?bpa=ebpi&t={quote(t)}"
    return fetch_json(url)


def pick_actual_redid(redactions_json: Dict) -> Optional[int]:
    reds = redactions_json.get("redactions") or []
    for r in reds:
        if r.get("actual") is True and r.get("hascontent") is True:
            return int(r["redid"])
    for r in reds:
        if r.get("hascontent") is True:
            return int(r["redid"])
    return None


def api_getcontent(redid: int) -> Dict:
    url = f"{ACTUAL_API}getcontent/?bpa=ebpi&rdk={redid}"
    return fetch_json(url)


def api_redtext_content(redid: int, ttl: int = 1) -> str:
    url = f"{ACTUAL_API}redtext?bpa=ebpi&t={redid}&ttl={ttl}"
    data = fetch_json(url)
    return data.get("redtext") or ""


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
        return []
    return [it for it in data if isinstance(it, dict) and it.get("unit") == "статья"]


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
    
    # Получаем текст, сохраняя структуру
    text = soup.get_text("\n", strip=False)
    
    # Удаляем id="..." из начала текста
    text = re.sub(r'^id="[^"]*"\s*', '', text)
    
    parts = []
    for line in text.splitlines():
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

    article_records: List[Dict] = []

    for d in docs:
        doc_hash = d["hash"]
        doc_title = d.get("doc_link_title", "")

        redit = None
        try:
            reds = api_redactions(doc_hash, ttl=1)
            redid = pick_actual_redid(reds)
            if not redid:
                logger.warning(f"[{code.upper()}] no redid for hash={doc_hash}")
                continue

            gc = api_getcontent(redid)
            nodes = iter_article_nodes_from_getcontent(gc)
            if not nodes:
                logger.warning(f"[{code.upper()}] no article nodes for hash={doc_hash}, redid={redid}")
                continue
                
            logger.info(f"[{code.upper()}] {doc_title[:60]} | redid={redid} | articles={len(nodes)}")

            rt = api_redtext_content(redid, ttl=1)
            if not rt:
                logger.warning(f"[{code.upper()}] no redtext content for hash={doc_hash}, redid={redid}")
                continue

            for node in nodes:
                if len(article_records) >= target * 3:  # safety buffer
                    break
                rec = build_article_record(code, doc_hash, redid, doc_title, node, rt)
                if rec:
                    article_records.append(rec)

            time.sleep(random.uniform(*SLEEP_RANGE))

        except Exception as e:
            logger.warning(f"[{code.upper()}] error for hash={doc_hash}: {e}")
            time.sleep(random.uniform(*SLEEP_RANGE))

    # Convert to pairs
    pairs = [article_record_to_pair(r) for r in article_records]

    # Clean + filter
    cleaned = []
    for p in pairs:
        case = clean_text(p.get("case", ""))
        article = clean_text(p.get("article", ""))
        if len(article) < MIN_CONTENT_LENGTH:
            continue
        if not case:
            continue
        cleaned.append({"case": case, "article": article})

    cleaned = dedupe_pairs(cleaned)

    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(cleaned)
    return cleaned[:target]

# -----------------------------
# Main
# -----------------------------
def main():
    stats = {}

    for code in CODES:
        logger.info("=" * 70)
        pairs = scrape_actual_code_pairs(code, TARGET_PER_CODE)
        stats[code] = len(pairs)

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
