import json
import re
from pathlib import Path

"""
Fixed and consolidated labeling script that replaces both auto_label_by_filename.py and labling_clean_data.py
"""

# Настройки
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

print("Текущая директория:", Path.cwd())
print("Путь к скрипту:", Path(__file__).parent.resolve())
print("Корень проекта:", PROJECT_ROOT)
print("Путь к данным:", DATA_DIR)
print("Существует ли DATA_DIR:", DATA_DIR.exists())

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Создана директория: {DATA_DIR}")

# Loading Dataset - Updated to use correct path
try:
    with open(DATA_DIR / "processed" / "synthetic_qa_cleaned.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {DATA_DIR / 'processed' / 'synthetic_qa_cleaned.json'}")
    print("Проверьте, что файл существует и путь указан правильно")
    exit(1)

law_mapping = {
    "НК": "Налоговый кодекс РФ",
    "ГК": "Гражданский кодекс РФ",
    "ТК": "Трудовой кодекс РФ",
    "ЖК": "Жилищный кодекс РФ",
    "КоАП": "Кодекс об административных правонарушениях",
    "УК": "Уголовный кодекс РФ",
    "ФЗ": "Федеральный закон",
}

# Add labels
for item in data:
    # Extract law code from article title
    article = item.get("article", item.get("article_title", ""))
    
    # Retrieve code of law
    match = re.match(r"Статья\s+\d+\s+([А-ЯA-Z]+)", article, re.IGNORECASE)
    if match:
        code = match.group(1).upper()
        full_name = law_mapping.get(code, "Другой закон")
    else:
        # Try to find law code in other ways
        # Check if it's in the title field
        title = item.get("title", item.get("article_title", ""))
        title_match = re.match(r"Статья\s+\d+\s+([А-ЯA-Z]+)", title, re.IGNORECASE)
        if title_match:
            code = title_match.group(1).upper()
            full_name = law_mapping.get(code, "Другой закон")
        else:
            full_name = "Неизвестно"
            code = "UNKNOWN"

    item["law"] = full_name
    item["law_code"] = code if code != "UNKNOWN" else "UNKNOWN"

# Saved with labels - Ensure output directory exists
if not (DATA_DIR / "processed").exists():
    (DATA_DIR / "processed").mkdir(exist_ok=True)

with open(DATA_DIR / "processed" / "synthetic_qa_labeled.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Готово! Добавлено {len(data)} меток")
print(f"Файл сохранен: {DATA_DIR / 'processed' / 'synthetic_qa_labeled.json'}")
print("\nПримеры разметки:")
for i, item in enumerate(data[:5], 1):
    title = item.get("article_title", item.get("title", "Без заголовка"))
    law = item.get("law", "Неизвестно")
    law_code = item.get("law_code", "???")
    print(f"{title} → {law} ({law_code})")