#!/usr/bin/env python3
"""
Загрузка и тест Saiga Mistral 7B через Hugging Face Transformers
"""

import os
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

# === ОСНОВНОЙ КОД ===
if __name__ == "__main__":
    check_and_download_model()
    
    model, tokenizer = load_model()
    
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
