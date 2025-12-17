#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------- Paths (relative) ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
REPORT_DIR = DATA_DIR / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Config ----------------
RANDOM_SEED = 42
MIN_ARTICLE_LEN = 200
MAX_ARTICLE_LEN = 8000
TARGET_PER_CODE = 300
QUESTIONS_PER_ARTICLE = 2

SYSTEM_PROMPT = "Ты эксперт по российскому законодательству. Отвечай только текстом статьи без добавлений."

FILES = {
    "actual_tk_150.json": "Трудового кодекса РФ",
    "actual_nk_150.json": "Налогового кодекса РФ",
    "actual_gk_150.json": "Гражданского кодекса РФ",
    "actual_koap_150.json": "Кодекса РФ об административных правонарушениях",
    # "actual_jk_150.json": "Жилищного кодекса РФ",
}


# ---------------- Utils ----------------
def clean_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_article_number_spacing(s: str) -> str:
    # "Статья 5. 57" -> "Статья 5.57"
    return re.sub(r"(Статья\s+\d+)\.\s+(\d+)", r"\1.\2", s)


def extract_article_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    if not title:
        return None, None
    t = normalize_article_number_spacing(clean_text(title))
    m = re.match(r"^Статья\s+(\d+(?:\.\d+)*)\.?\s*(.*)$", t)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def validate_article_text(article: str) -> bool:
    if not article:
        return False
    a = clean_text(article)
    if len(a) < MIN_ARTICLE_LEN:
        return False
    if len(a) > MAX_ARTICLE_LEN:
        return False
    low = a.lower()
    if "undefined" in low or "null" in low:
        return False
    if a.count(".") + a.count(";") < 2:
        return False
    return True


def truncate_article(article: str, max_len: int) -> str:
    a = article
    if len(a) <= max_len:
        return a
    cut = a[:max_len]
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind(";"))
    if last != -1 and last > int(max_len * 0.8):
        return cut[: last + 1].strip()
    return cut.strip()


def infer_code_key(code_name: str) -> str:
    cn = code_name.lower()
    if "административ" in cn:
        return "koap"
    if "граждан" in cn:
        return "gk"
    if "налог" in cn:
        return "nk"
    if "труд" in cn:
        return "tk"
    if "жилищ" in cn:
        return "jk"
    return "unknown"


def generate_retrieval_questions(article_num: str, code_name: str) -> List[str]:
    return [
        f"Приведи текст статьи {article_num} из {code_name}.",
        f"Текст статьи {article_num} {code_name}.",
        f"Какой текст у статьи {article_num} в {code_name}?",
    ]


def split_by_article(rows: List[Dict], test_ratio: float = 0.1, seed: int = 42):
    rnd = random.Random(seed)

    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        code = r["meta"]["code"]
        num = r["meta"]["article_number"]
        groups.setdefault((code, num), []).append(r)

    keys = list(groups.keys())
    rnd.shuffle(keys)

    n_test = max(1, int(len(keys) * test_ratio)) if keys else 0
    test_keys = set(keys[:n_test])

    train, test = [], []
    for k, items in groups.items():
        (test if k in test_keys else train).extend(items)

    return train, test, len(keys), n_test


# ---------------- Main builder ----------------
def build_dataset() -> None:
    rnd = random.Random(RANDOM_SEED)

    removed = []
    per_code_records: Dict[str, List[Dict]] = {}

    # 1) Load raw and normalize into unique articles
    for fname, code_name in FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            removed.append({"file": fname, "reason": "missing_file", "path": str(path)})
            continue

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            removed.append({"file": fname, "reason": "root_not_list"})
            continue

        code_key = infer_code_key(code_name)

        best_by_article = {}  # (code, num) -> rec

        for i, item in enumerate(raw):
            case_raw = item.get("case", "")
            article_raw = item.get("article", "")

            num, title_rest = extract_article_from_title(case_raw)
            if not num:
                removed.append({"file": fname, "row": i, "reason": "cannot_parse_article_number", "case": case_raw})
                continue

            article_text = truncate_article(clean_text(article_raw), MAX_ARTICLE_LEN)
            if not validate_article_text(article_text):
                removed.append({"file": fname, "row": i, "reason": "bad_article_text", "article_len": len(article_text)})
                continue

            key = (code_key, num)
            rec = {
                "code": code_key,
                "code_name": code_name,
                "article_number": num,
                "article_title": title_rest or "",
                "article": article_text,
            }

            if key not in best_by_article or len(rec["article"]) > len(best_by_article[key]["article"]):
                best_by_article[key] = rec

        articles = list(best_by_article.values())
        rnd.shuffle(articles)
        per_code_records[code_key] = articles

    # 2) Build dataset (case/article pairs)
    dataset: List[Dict] = []

    for code_key, articles in per_code_records.items():
        if not articles:
            continue

        need_articles = max(1, TARGET_PER_CODE // QUESTIONS_PER_ARTICLE)
        selected = articles[:need_articles]

        for art in selected:
            q_pool = generate_retrieval_questions(art["article_number"], art["code_name"])
            rnd.shuffle(q_pool)
            q_pool = q_pool[:QUESTIONS_PER_ARTICLE]

            for q in q_pool:
                dataset.append({
                    "case": q,
                    "article": art["article"],
                    "meta": {
                        "code": art["code"],
                        "code_name": art["code_name"],
                        "article_number": art["article_number"],
                        "system": SYSTEM_PROMPT
                    }
                })

    # 3) Final cap per code
    by_code_rows: Dict[str, List[Dict]] = {}
    for row in dataset:
        by_code_rows.setdefault(row["meta"]["code"], []).append(row)

    balanced: List[Dict] = []
    for code, rows in by_code_rows.items():
        rnd.shuffle(rows)
        balanced.extend(rows[:TARGET_PER_CODE])

    rnd.shuffle(balanced)

    # 4) Split train/test without leakage
    train, test, n_groups, n_test_groups = split_by_article(balanced, test_ratio=0.1, seed=RANDOM_SEED)

    # 5) Save outputs
    out_all = OUT_DIR / "case_article_dataset.json"
    out_train = OUT_DIR / "case_article_train.json"
    out_test = OUT_DIR / "case_article_test.json"
    removed_path = REPORT_DIR / "case_article_removed.json"
    summary_path = REPORT_DIR / "case_article_summary.json"

    out_all.write_text(json.dumps(balanced, ensure_ascii=False, indent=2), encoding="utf-8")
    out_train.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    out_test.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    removed_path.write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")

    out_per_code = {c: len(v) for c, v in by_code_rows.items()}
    summary = {
        "raw_dir": str(RAW_DIR),
        "files": list(FILES.keys()),
        "target_per_code": TARGET_PER_CODE,
        "questions_per_article": QUESTIONS_PER_ARTICLE,
        "min_article_len": MIN_ARTICLE_LEN,
        "max_article_len": MAX_ARTICLE_LEN,
        "total_out": len(balanced),
        "out_per_code_before_cap": out_per_code,
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_article_groups": n_groups,
        "test_article_groups": n_test_groups,
        "removed_rows": len(removed),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved ALL:   {out_all} ({len(balanced)} rows)")
    print(f"Saved TRAIN: {out_train} ({len(train)} rows)")
    print(f"Saved TEST:  {out_test} ({len(test)} rows)")
    print(f"Saved removed: {removed_path} ({len(removed)} rows)")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    build_dataset()
