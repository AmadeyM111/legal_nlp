#!/usr/bin/env python3
"""
Скрипт обработки собранных юридических данных
Очистка → Разделение 80/20 → Аугментация → Токенизация
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Загрузка ресурсов NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    russian_stopwords = set(stopwords.words('russian'))
except:
    logger.warning("NLTK resources not available, continuing without advanced preprocessing")
    russian_stopwords = set()

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_accordion(text: str) -> str:
    """Удаление элементов аккордеона и других визуальных элементов"""
    # Удаляем типичные элементы аккордеона
    text = re.sub(r'\s*\[.*?\]\s*', ' ', text)  # Удаляем скобки с содержимым
    text = re.sub(r'Закрыть\s+[^\n]*', '', text)  # Удаляем "Закрыть" и последующий текст
    text = re.sub(r'Подробнее\s+', '', text)  # Удаляем "Подробнее"
    text = re.sub(r'Показать\s+', '', text)  # Удаляем "Показать"
    text = re.sub(r'Скрыть\s+', '', text)  # Удаляем "Скрыть"
    text = re.sub(r'<[^>]+>', '', text)  # Удаляем HTML теги
    text = re.sub(r'→|←|↑|↓|▶|▼|►|◄', '', text)  # Удаляем стрелки
    text = re.sub(r'🔗|📌|✅|❌|⭐|✨|⚡', '', text)  # Удаляем эмодзи-иконки
    return text

def remove_emoji(text: str) -> str:
    """Удаление эмодзи и специальных символов"""
    # Удаляем эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_advanced(text: str) -> str:
    """Расширенная очистка текста"""
    # Удаляем ссылки
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Удаляем email
    text = re.sub(r'\S+@\S+', '', text)
    # Удаляем телефонные номера
    text = re.sub(r'\+?7[0-9\-\(\)\s]{10,}', '', text)
    # Удаляем даты в формате DD.MM.YYYY
    text = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}', '', text)
    # Удаляем числа (опционально, можно закомментировать если числа важны)
    # text = re.sub(r'\d+', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_raw_data() -> List[Dict]:
    """Загрузка всех сырых данных из директории"""
    all_data = []
    
    # Собираем все JSON файлы из директории raw
    for json_file in RAW_DATA_DIR.glob("*.json"):
        logger.info(f"Загрузка данных из: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Добавляем информацию о кодексе к каждому элементу
            code = json_file.stem.split('_')[0]  # извлекаем код кодекса из имени файла (например, 'gk' из 'gk_100.json')
            for item in data:
                item['code'] = code  # добавляем код кодекса как метку
            
            all_data.extend(data)
            logger.info(f"Загружено {len(data)} статей из {json_file.name}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {json_file.name}: {e}")
    
    logger.info(f"Всего загружено {len(all_data)} статей из всех кодексов")
    return all_data

def clean_data(raw_data: List[Dict]) -> List[Dict]:
    """Очистка данных - удаление мусора"""
    cleaned_data = []
    
    for item in raw_data:
        # Применяем все функции очистки
        cleaned_content = clean_accordion(item['content'])
        cleaned_content = remove_emoji(cleaned_content)
        cleaned_content = clean_text_advanced(cleaned_content)
        
        # Обновляем содержимое в элементе
        cleaned_item = item.copy()
        cleaned_item['content'] = cleaned_content
        
        # Добавляем только если контент не пустой после очистки
        if cleaned_content.strip():
            cleaned_data.append(cleaned_item)
    
    logger.info(f"После очистки осталось {len(cleaned_data)} статей из {len(raw_data)}")
    return cleaned_data

def split_by_codes(data: List[Dict], test_size: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """Разделение данных по кодексам с сохранением пропорций"""
    # Группируем данные по кодексам
    code_groups = {}
    for item in data:
        code = item.get('code', 'unknown')
        if code not in code_groups:
            code_groups[code] = []
        code_groups[code].append(item)
    
    logger.info(f"Найдены кодексы: {list(code_groups.keys())}")
    
    train_data = []
    test_data = []
    
    for code, group in code_groups.items():
        logger.info(f"Разделение {len(group)} статей для кодекса {code}")
        
        # Извлекаем только содержимое для стратификации (хотя в данном случае стратификация будет по кодексам)
        contents = [item['content'] for item in group]
        
        # Разделяем данные
        train_contents, test_contents = train_test_split(
            list(zip(group, contents)), 
            test_size=test_size, 
            random_state=42
        )
        
        # Восстанавливаем полные элементы данных
        train_group = [item for item, _ in train_contents]
        test_group = [item for item, _ in test_contents]
        
        train_data.extend(train_group)
        test_data.extend(test_group)
        
        logger.info(f"  - Обучающая выборка: {len(train_group)}")
        logger.info(f"  - Тестовая выборка: {len(test_group)}")
    
    logger.info(f"Итого: обучение {len(train_data)}, тест {len(test_data)}")
    return train_data, test_data

def augment_data(data: List[Dict]) -> List[Dict]:
    """Аугментация данных (синонимы, перефразировки)"""
    augmented_data = []
    
    for item in data:
        augmented_data.append(item)  # Добавляем оригинальный элемент
        
        # Простая аугментация - перефразировка заголовков
        original_title = item['title']
        
        # Создаем варианты заголовка
        variations = [
            # Убираем "действующая редакция" из заголовка
            re.sub(r'\s+\(действующая редакция\)', '', original_title),
            # Заменяем "статья" на "положение"
            re.sub(r'^Статья (\d+)', r'Положение \1', original_title),
            # Убираем нумерацию из заголовка
            re.sub(r'^Статья \d+\s*', '', original_title),
        ]
        
        # Добавляем варианты только если они отличаются от оригинала
        for variation in variations:
            if variation != original_title and variation.strip():
                augmented_item = item.copy()
                augmented_item['title'] = variation
                augmented_item['augmentation_type'] = 'title_variation'
                augmented_data.append(augmented_item)
    
    logger.info(f"Аугментация: {len(data)} -> {len(augmented_data)} элементов")
    return augmented_data

def tokenize_data(data: List[Dict], model_name: str = "IlyaGusev/saiga_mistral_7b") -> List[Dict]:
    """Токенизация данных"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        # Если модель не найдена, используем базовый токенизатор
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    tokenized_data = []
    
    for item in data:
        # Создаем текст для токенизации (объединяем заголовок и содержимое)
        text = f"{item['title']}\n\n{item['content']}"
        
        # Токенизируем
        tokens = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=2048,  # ограничиваем длину
            return_tensors=None
        )
        
        # Добавляем токенизированный элемент
        tokenized_item = item.copy()
        tokenized_item['input_ids'] = tokens['input_ids']
        tokenized_item['attention_mask'] = tokens.get('attention_mask', [1] * len(tokens['input_ids']))
        tokenized_item['token_count'] = len(tokens['input_ids'])
        
        tokenized_data.append(tokenized_item)
    
    logger.info(f"Токенизация завершена, среднее количество токенов: {sum(item['token_count'] for item in tokenized_data) / len(tokenized_data):.2f}")
    return tokenized_data

def save_processed_data(train_data: List[Dict], test_data: List[Dict]):
    """Сохранение обработанных данных"""
    # Сохраняем обучающую выборку
    train_file = PROCESSED_DATA_DIR / "train_data.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # Сохраняем тестовую выборку
    test_file = PROCESSED_DATA_DIR / "test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Обработанные данные сохранены:")
    logger.info(f"  - Обучающая выборка: {train_file}")
    logger.info(f"  - Тестовая выборка: {test_file}")
    logger.info(f"  - Обучающих примеров: {len(train_data)}")
    logger.info(f"  - Тестовых примеров: {len(test_data)}")

def main():
    """Основная функция обработки данных"""
    logger.info("Начало обработки данных")
    
    # 1. Загрузка сырых данных
    logger.info("#1 Загрузка сырых данных")
    raw_data = load_raw_data()
    
    if not raw_data:
        logger.error("Не найдено сырых данных для обработки")
        return
    
    # 2. Очистка данных
    logger.info("#2 Очистка данных (удаление мусора)")
    cleaned_data = clean_data(raw_data)
    
    if not cleaned_data:
        logger.error("После очистки не осталось данных")
        return
    
    # 3. Разделение 80/20 по кодексам
    logger.info("#3 Разделение данных 80/20 по кодексам")
    train_data, test_data = split_by_codes(cleaned_data, test_size=0.2)
    
    # 4. Аугментация ТОЛЬКО обучающей выборки
    logger.info("#4 Аугментация обучающей выборки")
    augmented_train = augment_data(train_data)
    
    # 5. Токенизация
    logger.info("#5 Токенизация данных")
    tokenized_train = tokenize_data(augmented_train)
    tokenized_test = tokenize_data(test_data)
    
    # 6. Сохранение обработанных данных
    logger.info("#6 Сохранение обработанных данных")
    save_processed_data(tokenized_train, tokenized_test)
    
    logger.info("Обработка данных завершена успешно!")

if __name__ == "__main__":
    main()