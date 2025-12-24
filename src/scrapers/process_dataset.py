import json
import re
from pathlib import Path

def clean_text(text: str) -> str:
    text = re.sub(r"\(В редакции.*?\)", "", text)
    text = re.sub(r"\(Дополнение.*?\)", "", text)
    text = re.sub(r"\(Утратила силу.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "apk_rf_articles.json": "Административно-процессуальный кодекс РФ",
    "tk_rf_articles.json": "Трудовой кодекс РФ",
    "gk_rf_articles.json": "Гражданский кодекс РФ",
    "nk_rf_articles.json": "Налоговый кодекс РФ",
    "koap_rf_articles.json": "Кодекс РФ об административных правонарушениях",
}

all_data = []

for file_name, code_name in files.items():
    file_path = RAW_DIR / file_name
    if not file_path.exists():
        print(f"Пропущен: {file_path}")
        continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        raw_case = item["title"]
        raw_article = item["content"]
        
        case = clean_text(raw_case)
        article = clean_text(raw_article)
        
        # Извлекаем номер статьи из case (обычно начинается с "Статья XX")
        match = re.match(r"Статья\s+([\d½\-\s]+)\.?\s*(.+)", case)
        if match:
            article_num = match.group(1).strip()
            article_title = match.group(2).strip()  # Описание статьи после "Статья N."
        else:
            article_num = "неизвестная"
            article_title = case
        
        # Создаем короткий вопрос для пользователя на основе заголовка статьи
        # Берем только первые 5-6 слов для краткости
        title_words = article_title.split()[:6]  # Берем только первые 6 слов
        short_description = ' '.join(title_words)
        
        user_query = f"Какая статья из {code_name} регулирует: {short_description}"
        
        # Для assistant ответа используем полный текст статьи, но без дублирования заголовка
        # Если article начинается с заголовка статьи, убираем его
        if article.startswith(case):
            # Убираем заголовок статьи из начала текста
            assistant_answer = article[len(case):].strip()
            if assistant_answer.startswith(". "):
                assistant_answer = assistant_answer[2:].strip()
        else:
            # Если article не начинается с заголовка, используем его полностью
            assistant_answer = article
        
        # Формируем ответ ассистента: "Статья N. Полный текст статьи"
        assistant_response = f"Статья {article_num}. {article_title}"
        if assistant_answer and assistant_answer.strip() != article_title.strip():
            # Добавляем основной текст статьи, если он отличается от заголовка
            assistant_response += " " + assistant_answer
        
        # Проверяем, что у нас есть хотя бы минимальные данные
        if article_title:
            all_data.append({
                "messages": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

output_path = PROCESSED_DIR / "all_codes_fixed_qlora.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"Готово: {len(all_data)} примеров → {output_path}")