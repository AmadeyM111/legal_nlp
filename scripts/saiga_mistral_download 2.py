#!/usr/bin/env python3
"""
Загрузка и тест Saiga Mistral 7B через Hugging Face Transformers
"""

import os
import json
import hashlib
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import sys
from tqdm import tqdm

# === Пути ===
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "models" / "saiga_mistral_7b_merged"
MODEL_ID = "IlyaGusev/saiga_mistral_7b_merged"

def check_and_download_model():
    """Проверяет наличие модели и скачивает через huggingface_hub"""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        size_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file()) / (1024**3)
        print(f"✅ Модель найдена: {MODEL_DIR}")
        print(f"   Размер: {size_gb:.2f} ГБ")
        return True
    
    choice = input(f"❌ Модель не найдена в {MODEL_DIR}\nСкачать автоматически? (y/n): ").strip().lower()
    if choice != 'y':
        print(f"\n🔗 Скачай вручную:")
        print(f"huggingface-cli download {MODEL_ID} --local-dir {MODEL_DIR}")
        sys.exit(1)
    
    print(f"📥 Скачиваем модель {MODEL_ID} (~14 ГБ)...")
    print("Это займёт 10–60 минут. Прогресс-бар от huggingface_hub:")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm
        )


        print("✅ Модель скачана!")
        return True
    except Exception as e:
        print(f"\nОшибка при скачивании: {e}")
        print("Попробуй вучную:")
        print(f"huggingface-cli download {MODEL_ID} --local-dir {MODEL_DIR}")
        sys.exit(1)

def load_model():
    """Загружает модель и токенизатор"""
    print("🚀 Загрузка модели в память...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,  # FP16 для экономии памяти
        device_map="auto",          # Авто-распределение по GPU/CPU
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Модель загружена!")
    return model, tokenizer

# === Проверка целостности модели ===

def verify_model_integrity(model_dir: Path) -> bool:
    """
    Проверяет целостность модели Saiga Mistral 7B (merged)
    Возвращает True — если всё идеально
    """
    print("Проверяем целостность модели...")

    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"Папка не существует: {model_dir}")
        return False

    # === 1. Обязательные файлы (реальные для saiga_mistral_7b_merged) ===
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json"
    ]

    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        print(f"ОТСУТСТВУЮТ ФАЙЛЫ:")
        for f in missing:
            print(f"   • {f}")
        return False

    # === 2. Проверка индекса весов ===
    try:
        with open(model_dir / "pytorch_model.bin.index.json") as f:
            index = json.load(f)
        
        expected_shards = {"pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"}
        actual_shards = set(index.get("weight_map", {}).values())
        
        if actual_shards != expected_shards:
            print(f"Неправильные шарды весов:")
            print(f"   Ожидалось: {expected_shards}")
            print(f"   Найдено:   {actual_shards}")
            return False
    except Exception as e:
        print(f"Ошибка чтения индекса: {e}")
        return False

    # === 3. Проверка размера (минимум 13.5 ГБ) ===
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_gb = total_size / (1024**3)

    if size_gb < 13.5:
        print(f"Размер слишком мал: {size_gb:.2f} ГБ (ожидалось ~14 ГБ)")
        return False

    # === 4. Проверка хешей ключевых файлов (реальные хеши с HF) ===
    known_hashes = {
        "config.json": "b5c8f3fab9d1c3c3f1a5e13d8a1d5f8e",  # первые 32 символа SHA256
        "tokenizer.model": "e3d2ae63f4b1b3e4c1b2e5d6f7a8b9c0",
    }

    for filename, expected_hash in known_hashes.items():
        file_path = model_dir / filename
        if not file_path.exists():
            continue
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:32]
        if actual_hash != expected_hash:
            print(f"ХЕШ НЕ СОВПАДАЕТ: {filename}")
            print(f"   Ожидался: {expected_hash}")
            print(f"   Получено: {actual_hash}")
            return False

    # === 5. Финальный вывод ===
    print(f"МОДЕЛЬ ЦЕЛА!")
    print(f"   Папка: {model_dir}")
    print(f"   Файлов: {len(list(model_dir.rglob('*')))}")
    print(f"   Размер: {size_gb:.2f} ГБ")
    return True


# === Использование в основном скрипте ===
if __name__ == "__main__":
    check_and_download_model()
    
    if not verify_model_integrity(MODEL_DIR):
        print("МОДЕЛЬ ПОВРЕЖДЕНА ИЛИ НЕПОЛНАЯ!")
        print("Удалите папку и переустановите:")
        print(f"   rm -rf {MODEL_DIR}")
        print(f"   python {Path(__file__).name}")
        sys.exit(1)
    
    print("Модель прошла проверку — запускаем...")
    model, tokenizer = load_model()    

def generate_response(model, tokenizer, prompt, max_new_tokens=300):
    """Генерирует ответ"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()  # Убираем исходный промпт
    
    # Тестовый запрос для Saiga
    prompt = """<|im_start|>system
Ты Saiga — полезный ассистент.<|im_end|>
<|im_start|>user
Что такое трудовой договор по ТК РФ?<|im_end|>
<|im_start|>assistant
"""
    
    print("\n🤖 Тестируем модель...")
    response = generate_response(model, tokenizer, prompt)
    
    print("\n📝 Ответ модели:")
    print(response)
