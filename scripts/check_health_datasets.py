#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Запуск через CLI:
python check_health_ds.py --train data/processed/train.jsonl --test data/processed/test.jsonl --val data/processed/val.jsonl --report-name ds_sanity_report.json
"""

import json
import re
import argparse
import logging
import hashlib
import statistics
from pathlib import Path
from typing import Dict, Set, Generator, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

NOISE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'<[^>]*>', r'id="[^"]*"', r'undefined', r'null', 
        r'\[object Object\]', r'Закрыть|Развернуть|Свернуть',
    ]
]

def stream_jsonl(filepath: Path) -> Generator[Tuple[int, Dict, str], None, None]:
    """Чтение файла с проверкой существования"""
    # Преобразуем в абсолютный путь для ясности в логах
    abs_path = filepath.resolve()
    
    if not filepath.exists():
        logger.error(f"ФАЙЛ НЕ НАЙДЕН: {abs_path}")
        return

    # Проверка на пустой файл
    if filepath.stat().st_size == 0:
        logger.warning(f"ФАЙЛ ПУСТ: {abs_path}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            is_json_array = (first_char == '[')
            
            if is_json_array:
                logger.info(f"Обнаружен JSON Array формат в {filepath.name}")
                data = json.load(f)
                for i, item in enumerate(data, 1):
                    yield i, item, json.dumps(item, ensure_ascii=False)
            else:
                # JSONL
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    try:
                        yield i, json.loads(line), line
                    except json.JSONDecodeError:
                        logger.warning(f"Ошибка парсинга строки {i} в {filepath.name}")
                        continue
    except Exception as e:
        logger.error(f"Ошибка чтения {abs_path}: {e}")

def get_content_hash(item: Dict) -> str:
    messages = item.get("messages", [])
    if not isinstance(messages, list): return ""
    content_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            content_parts.append(f"{role}:{content}")
    full_text = "|".join(content_parts)
    return hashlib.md5(full_text.encode('utf-8')).hexdigest()

def analyze_dataset_stream(
    filepath: Path, 
    dataset_name: str, 
    reference_hashes: Set[str] = None
) -> Tuple[Dict, Set[str]]:
    
    logger.info(f"--- Анализ: {dataset_name.upper()} ---")
    logger.info(f"Путь: {filepath.resolve()}")
    
    stats = {
        "name": dataset_name,
        "total_records": 0,
        "valid_records": 0,
        "bad_format_count": 0,
        "duplicates_internal": 0,
        "leakage_count": 0,
        "noise_count": 0,
        "role_lengths": {"user": [], "assistant": [], "system": []},
        "issues_sample": []
    }
    
    local_hashes = set()
    
    for line_num, item, raw_line in stream_jsonl(filepath):
        stats["total_records"] += 1
        
        # 1. Проверка структуры
        messages = item.get("messages")
        if not messages or not isinstance(messages, list):
            stats["bad_format_count"] += 1
            if len(stats["issues_sample"]) < 10:
                stats["issues_sample"].append(f"Line {line_num}: Invalid 'messages' structure")
            continue

        # 2. Длины и Шум
        has_noise = False
        for msg in messages:
            if not isinstance(msg, dict): continue
            role = msg.get("role")
            content = str(msg.get("content", ""))
            
            if role in stats["role_lengths"]:
                stats["role_lengths"][role].append(len(content))
            
            if not has_noise:
                for pattern in NOISE_PATTERNS:
                    if pattern.search(content):
                        has_noise = True
                        break
        
        if has_noise: stats["noise_count"] += 1

        # 3. Дедупликация
        curr_hash = get_content_hash(item)
        if not curr_hash: continue
            
        if curr_hash in local_hashes:
            stats["duplicates_internal"] += 1
        else:
            local_hashes.add(curr_hash)
            stats["valid_records"] += 1
            
        if reference_hashes and curr_hash in reference_hashes:
            stats["leakage_count"] += 1

    # Агрегация длин
    final_lengths = {}
    for role, lengths in stats["role_lengths"].items():
        if lengths:
            final_lengths[role] = {
                "min": min(lengths),
                "max": max(lengths),
                "median": int(statistics.median(lengths)),
                "count": len(lengths)
            }
    stats["length_stats"] = final_lengths
    del stats["role_lengths"]

    if stats["total_records"] == 0:
        logger.warning(f"⚠️ Датасет {dataset_name} пуст или не прочитан!")
    else:
        logger.info(f"ОК. Прочитано: {stats['total_records']}")

    return stats, local_hashes

def main():
    # 1. Определяем, где лежит этот скрипт
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    
    # 2. Определяем корень проекта. 
    # Если скрипт в /Users/.../experimental/scripts/
    # То корень  в /Users/.../experimental/
    project_root = script_dir.parent 

    # 3. Формируем пути к данным ОТ КОРНЯ ПРОЕКТА
    default_data_dir = project_root / "data"
    
    default_train = default_data_dir / "processed" / "train.jsonl"
    default_test  = default_data_dir / "processed" / "test.jsonl"
    default_val   = default_data_dir / "processed" / "val.jsonl"
    
    # Папка для отчетов
    reports_dir = default_data_dir / "reports"

    # Парсер аргументов
    parser = argparse.ArgumentParser()
    
    # Теперь default значения — это АБСОЛЮТНЫЕ пути, вычисленные выше
    parser.add_argument("--train", type=Path, default=default_train)
    parser.add_argument("--test", type=Path, default=default_test)
    parser.add_argument("--val", type=Path, default=default_val)
    parser.add_argument("--report-name", type=str, default="ds_sanity_report.json")
    
    args = parser.parse_args()

    # Гарантируем, что папка существует (exist_ok=True не даст ошибку, если папка есть)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Полный путь к отчету
    report_path = reports_dir / args.report_name
    full_report = {"datasets": {}}
    
 # Анализ
    train_stats, train_hashes = analyze_dataset_stream(args.train, "train")
    full_report["datasets"]["train"] = train_stats

    test_stats, test_hashes = analyze_dataset_stream(args.test, "test", reference_hashes=train_hashes)
    full_report["datasets"]["test"] = test_stats
    
    # Val проверяем только если файл существует
    if args.val.exists():
        combined = train_hashes.union(test_hashes)
        val_stats, _ = analyze_dataset_stream(args.val, "val", reference_hashes=combined)
        full_report["datasets"]["val"] = val_stats
    else:
        logger.info(f"Файл val не найден, пропускаем: {args.val}")

    # Сохранение
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ ОТЧЕТ СОХРАНЕН В: {report_path.resolve()}")
    
    # Краткая таблица в консоль
    print(f"\n{'DATASET':<10} {'TOTAL':<10} {'VALID':<10} {'DUPES':<10} {'LEAKAGE':<10} {'BAD_FMT':<10}")
    print("-" * 65)
    for name, s in full_report["datasets"].items():
        print(f"{name.upper():<10} {s['total_records']:<10} {s['valid_records']:<10} "
              f"{s['duplicates_internal']:<10} {s['leakage_count']:<10} {s['bad_format_count']:<10}")
    print("-" * 65)

if __name__ == "__main__":
    main()
