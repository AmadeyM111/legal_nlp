#!/usr/bin/env python3
"""
Saiga Mistral 7B — полностью автономная загрузка и запуск
"""

from pathlib import Path
from llama_cpp import Llama
import requests
from tqdm import tqdm
import sys

# === ПУТИ ===
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "models" / "saiga"
MODEL_PATH = MODEL_DIR / "model-q4_K.gguf"
URL = "https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf"

def download_with_progress():
    print(f"Скачиваем модель Saiga Mistral 7B (~4.8 ГБ)...")
    print("Это займёт 5–40 минут. Прогресс-бар ниже:")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(MODEL_PATH, "wb") as f:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Saiga 7B",
            ncols=90,
            colour="cyan",
            leave=True,           # ← оставляет бар после завершения
            mininterval=1.0,      # ← обновляет не чаще раза в секунду
            maxinterval=2.0,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"\nМодель успешно скачана!")
    print(f"Путь: {MODEL_PATH}")

# === Автоматическая загрузка ===
if not MODEL_PATH.exists():
    download_with_progress()
else:
    print(f"Модель уже есть: {MODEL_PATH}")

# === Загрузка и тест ===
print("\nЗагружаем модель в память...")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=8, verbose=False)

print("Готово! Задай вопрос:")
while True:
    query = input("\nТы: ")
    if query.lower() in ["выход", "exit", "quit"]:
        break
    
    output = llm(query, max_tokens=500, temperature=0.3, stop=["\n\n"])
    print(f"Saiga: {output['choices'][0]['text']}")


    def download_model():
        print("\nСкачиваем модель Saiga Mistral 7B (~4.8 ГБ)...")
        print("Прогресс-бар ниже — одна строка, обновляется в реальном времени")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))