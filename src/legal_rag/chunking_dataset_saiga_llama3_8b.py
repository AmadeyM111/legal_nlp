import json
from transformers import AutoTokenizer
from pathlib import Path
from datasets import load_dataset

# Настройки
MODEL_ID = "IlyaGusev/saiga_llama3_8b"
MAX_TOKENS = 7500  # Оставляем запас до 8192 для системного промпта и вопроса
OVERLAP = 500      # Сколько токенов из конца предыдущего чанка берем в начало следующего
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_codes_fixed_qlora.json"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
train_dataset = load_dataset("json", data_files=DATA_PATH.as_posix(), split="train")

def chunk_text_by_tokens(text, max_tokens, overlap, tokenizer):
    """Разбивает текст на куски по токенам с перекрытием."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        
        if end >= len(tokens):
            break
        start += (max_tokens - overlap)
    
    return chunks

# Загрузка
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_dataset = []
chunked_count = 0

print(f"🚀 Обработка {len(data)} примеров...")

for ex in data:
    user_message = ex["messages"][1]      # Запрос пользователя
    assistant_content = ex["messages"][-1]["content"] # Длинный ответ
    
    # Считаем токены только ответа
    tokens_count = len(tokenizer.encode(assistant_content))
    
    if tokens_count > MAX_TOKENS:
        # Разбиваем на части
        text_chunks = chunk_text_by_tokens(assistant_content, MAX_TOKENS, OVERLAP, tokenizer)
        
        for i, chunk in enumerate(text_chunks):
            # Создаем новый пример для каждого чанка
            new_ex = {
                "messages": [
                    ex["messages"][0], # Системный промпт
                    {
                        "role": "user", 
                        "content": f"{user_message['content']} (Часть {i+1}/{len(text_chunks)})"
                    },
                    {"role": "assistant", "content": chunk}
                ]
            }
            new_dataset.append(new_ex)
        chunked_count += 1
    else:
        new_dataset.append(ex)

# Сохранение
with open(DATA_PATH.as_posix(), 'w', encoding='utf-8') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Готово!")
print(f"Обработано длинных статей: {chunked_count}")
print(f"Итоговое количество примеров: {len(new_dataset)} (было {len(data)})")
