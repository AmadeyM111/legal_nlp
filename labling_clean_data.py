from pathlib import Path

# Fix PROJECT_ROOT - it should point to the main project directory, not src
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Go up from src to project root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

print("Текущая директория:", Path.cwd())
print("Путь к скрипту:", Path(__file__).parent.resolve())
print("Корень проекта:", PROJECT_ROOT)
print("Путь к данным:", DATA_DIR)
print("Существует ли DATA_DIR:", DATA_DIR.exists())

# Loading Dataset - Correct path to processed data
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
"УК": "Уголовный кодекс РФ",  # Fixed typo
"ФЗ": "Федеральный закон",
}

# Add labels
for item in data:
article = item["article"]

# Retrieve code of law
match = re.match(r"Статья\s+\d+\s+([А-ЯA-Z]+)", article, re.IGNORECASE)
if match:
code = match.group(1).upper()
full_name = law_mapping.get(code, "Другой закон")
else:
full_name = "Неизвестно"

item["law"] = full_name
item["law_code"] = code if match else "UNKNOWN"

# Saved with labels - Ensure output directory exists
if not DATA_DIR.exists():
DATA_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "processed" / "synthetic_qa_labeled.json", "w", encoding="utf-8") as f:
json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Готово! Добавлено {len(data)} меток")
print("Примеры:")
for item in data[:5]:
print(f"{item['article']} → {item['law']}")