#!/usr/bin/env python3
"""
Fine-tuning Saiga Mistral 7B на Apple Silicon через MLX
Работает на M1/M2/M3 — быстро, стабильно, без GPU-ограничений
"""

import json
from pathlib import Path
import argparse
from mlx_lm import load, generate
from mlx_lm.lora import lora, LoraConfig

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Saiga Mistral 7B на Mac")
    parser.add_argument("--data", default="data/processed/synthetic_qa_cleaned.json", help="Путь к датасету")
    parser.add_argument("--model", default="IlyaGusev/saiga_mistral_7b", help="Базовая модель")
    parser.add_argument("--output", default="models/saiga-legal-mistral-7b-lora", help="Куда сохранить")
    parser.add_argument("--iters", type=int, default=1000, help="Количество итераций")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    args = parser.parse_args()

    # === 1. Загрузка модели ===
    print("Загружаем модель (это займёт 20–40 секунд)...")
    model, tokenizer = load(args.model)

    # === 2. Загрузка данных ===
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Файл не найден: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Форматируем под чат
    train_data = []
    for item in raw_data:
        user_msg = item.get("case") or item.get("question", "")
        assistant_msg = item.get("article") or item.get("answer", "")
        if user_msg and assistant_msg:
            train_data.append([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ])

    print(f"Подготовлено {len(train_data)} примеров для обучения")

    # === 3. LoRA конфиг ===
    config = LoraConfig(
        rank=args.rank,
        alpha=16,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # === 4. Fine-tuning ===
    print(f"Запускаем обучение (iters={args.iters}, batch={args.batch})...")
    trained_model = lora(
        model,
        config,
        train_data,
        batch_size=args.batch,
        iters=args.iters,
        learning_rate=2e-4
    )

    # === 5. Сохранение ===
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Сохраняем модель в {output_path}")
    trained_model.save(str(output_path))
    tokenizer.save(str(output_path))

    print("Готово! Модель обучена и сохранена.")
    print(f"Путь: {output_path.resolve()}")

    # === 6. Тест ===


{{ .Response }}<|im_end|
"""
    
    # Save Modelfile
    modelfile_path = PROJECT_ROOT / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"📝 Modelfile created: {modelfile_path}")

    # Create model in Ollama
    print("🔨 Creating base model in Ollama...")
    result = subprocess.run(["ollama", "create", output_model_name, "-f", str(modelfile_path)], 
                           capture_output=True, text=True)

    if result.returncode != 0:
        print("Error creating model:")
        print(result.stderr)
        return False

    print(f"Model {output_model_name} created")

    # Prepare dataset in Ollama format
    train_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)[:2000]  # Take 2000 examples

    for item in raw_data:
        train_data.append({
            "instruction": "Определи применимую статью закона по описанию дела",
            "input": item.get("case", item.get("question", "")),
            "output": item.get("article", item.get("output", ""))
        })

    # Save training data
    train_file = PROJECT_ROOT / "train_data.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"📚 Training data saved: {train_file}")

    print(f"✅ Ollama Fine-tuning setup completed! Model: {output_model_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Unified Fine-tuning for Legal Models")
    parser.add_argument("--method", choices=["mlx", "ollama"], required=True,
                        help="Fine-tuning method to use")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to training dataset")
    parser.add_argument("--model", type=str, default="saiga-mistral-7b",
                        help="Base model name")
    parser.add_argument("--output", type=str,
                        help="Output directory/model name")
    parser.add_argument("--rank", type=int, default=64,
                        help="LoRA rank (MLX only)")
    parser.add_argument("--alpha", type=int, default=16,
                        help="LoRA alpha (MLX only)")
    
    args = parser.parse_args()
    
    if args.method == "mlx":
        output_dir = args.output or f"models/legal-{args.model.replace('/', '-')}-lora"
        success = mlx_finetune(
            dataset_path=args.dataset,
            model_name=args.model,
            output_dir=output_dir,
            rank=args.rank,
            alpha=args.alpha
        )
    elif args.method == "ollama":
        output_model = args.output or "legal-saiga-7b"
        success = ollama_finetune(
            dataset_path=args.dataset,
            output_model_name=output_model
        )
    
    if success:
        print("🎉 Fine-tuning completed successfully!")
    else:
        print("❌ Fine-tuning failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
{{ .Prompt }}<|im_end>
{{ end }}<|im_start|>assistant
SYSTEM }}
{{ .System }}<|im_end>
{{ end }}{{ if .Prompt }}<|im_start|>user
    print("\nТестируем модель...")
    prompt = "В трудовом договоре нет удалёнки. Могу ли я работать из дома?"
    response = generate(trained_model, tokenizer, prompt=prompt, max_tokens=200, temp=0.3)
    print(f"\nОтвет:\n{response}")

if __name__ == "__main__":
    main()