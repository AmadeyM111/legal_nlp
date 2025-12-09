#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from datasets import Dataset

# ──────────────── PROJECT ROOT & SMART MODEL RESOLUTION ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # src/legal_rag/ → project root
MODELS_DIR = PROJECT_ROOT / "models"

def resolve_model_path(model_arg: str) -> str:
    """Умно определяет — это HF-репозиторий, локальная папка или относительный путь"""
    if model_arg.startswith(("http://", "https://", "hf://")):
        return model_arg
    if "/" in model_arg and model_arg.count("/") >= 1:  # выглядит как HF repo
        return model_arg
    
    # Иначе — считаем, что это имя папки внутри ./models/
    candidate = MODELS_DIR / model_arg
    if candidate.exists() and (candidate / "config.json").exists():
        return str(candidate.resolve())
    else:
        raise FileNotFoundError(
            f"Модель не найдена локально по пути: {candidate}\n"
            f"Проверь, что папка существует в {MODELS_DIR}\n"
            f"Или укажи полное имя на HF: IlyaGusev/saiga_mistral_7b_lora"
        )

# Автоопределение бэкенда и установка нужного
def install_backend():
    try:
        import mlx
        # Проверяем, есть ли safetensors файлы в модели
        import os
        model_path = resolve_model_path("saiga_mistral_7b_merged")
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        if safetensors_files:
            return "mlx"
        else:
            print("Формат safetensors не найден, используем PyTorch backend")
            return "cuda"  # fallback на PyTorch если нет safetensors
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cuda" # fallback

BACKEND = install_backend()

if BACKEND == "mlx":
    # Apple Silicon
    from mlx_lm import load, lora
    print("Apple Silicon detected → MLX-LM")

    def finetune(args):
        print(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)
        
        print(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Validate data format
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
        
        train_data = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item at index {i}")
                continue
                
            user_content = item.get("case") or item.get("question", "")
            assistant_content = item.get("article") or item.get("answer", "")
            
            if not user_content.strip() or not assistant_content.strip():
                print(f"Warning: Skipping item {i} with empty content")
                continue
                
            train_data.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])
        
        if len(train_data) == 0:
            raise ValueError("No valid training data found after filtering")
            
        print(f"Training with {len(train_data)} examples")
        
        model = lora(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            batch_size=args.batch,
            iters=args.iters,
            rank=args.rank,
            alpha=32,
            dropout=0.05,
            learning_rate=args.lr,
            target_modules="all-linear",
        )
        Path(args.output).mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        tokenizer.save(args.output)
        print(f"Model saved to {args.output}")

else:
    # Любой GPU/CPU — Transformers + PEFT + Unsloth (самый быстрый на CUDA в 2025)
    import torch
    from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    
    # Определяем, поддерживается ли bfloat16
    def is_bfloat16_supported():
        try:
            import torch
            # Проверяем поддержку bfloat16 на доступных устройствах
            if torch.cuda.is_available():
                # Для CUDA проверяем, поддерживает ли GPU bfloat16
                return torch.cuda.is_bf16_supported()
            else:
                # Для CPU и MPS поддержка bfloat16 может быть ограничена
                return False
        except:
            return False

    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
        print("Unsloth detected → до 2.5× быстрее + 70% меньше VRAM")
    except ImportError:
        USE_UNSLOTH = False
        print("Transformers + QLoRA")

    def finetune(args):
        print(f"Loading model: {args.model}")
        if USE_UNSLOTH:
            model, tokenizer = FastLanguageModel.from_pretrained(
                args.model,
                dtype=None,  # авто bf16/fp16
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.rank,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                use_gradient_checkpointing="unsloth",
            )
        else:
            from peft import prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Проверяем наличие bitsandbytes
            try:
                import bitsandbytes as bnb
                # Если bitsandbytes установлен, используем 4-bit квантование
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            except ImportError:
                # Если bitsandbytes не установлен, проверяем, есть ли safetensors
                try:
                    import safetensors
                    # Если safetensors установлен, грузим с device_map
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                        device_map="auto",
                    )
                except ImportError:
                    # Если safetensors не установлен, грузим без device_map
                    print("safetensors не установлен, загружаем модель без device_map для избежания offload ошибок")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                        # Убираем device_map, чтобы избежать ошибки offloading
                    )
                    # Ручное перемещение на GPU если доступно
                    if torch.cuda.is_available():
                        model = model.to('cuda')
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                r=args.rank,
                lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ))

        # Данные
        print(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Validate data format
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
            
        # Filter out invalid entries
        filtered_raw = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item at index {i}")
                continue
                
            user_content = item.get("case") or item.get("question", "")
            ans_content = item.get("article") or item.get("answer", "")
            
            if not user_content.strip() or not ans_content.strip():
                print(f"Warning: Skipping item {i} with empty content")
                continue
                
            filtered_raw.append(item)
            
        if len(filtered_raw) == 0:
            raise ValueError("No valid training data found after filtering")
            
        print(f"Processing {len(filtered_raw)} valid examples")
        
        dataset = Dataset.from_list(filtered_raw)
        def formatting_func(ex):
            user = ex.get("case") or ex.get("question", "")
            ans  = ex.get("article") or ex.get("answer", "")
            text = f"<s>[INST] {user} [/INST] {ans}</s>"
            return {"text": text}
        dataset = dataset.map(lambda x: {"text": formatting_func(x)["text"]})

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=max(1, 8//args.batch),  # Prevent division by zero
                num_train_epochs=3,
                learning_rate=args.lr,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                save_steps=500,
                output_dir=args.output,
                optim="adamw_8bit",
                report_to="none",
                remove_unused_columns=False,  # Important for custom formatting
            ),
            dataset_text_field="text",
            max_seq_length=2048,
        )
        trainer.train()
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Model saved to {args.output}")

# ──────────────────────── ARGUMENTS ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "synthetic_qa_cleaned.json")
parser.add_argument("--model", type=str, default="saiga_mistral_7b_merged",   # ← теперь просто имя папки!
                    help="Либо имя папки в ./models/, либо полное HF repo (IlyaGusev/...)")
parser.add_argument("--output", type=str, default="models/saiga-legal-7b")
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

# ← Вот и вся магия:
MODEL_PATH = resolve_model_path(args.model)
args.model = MODEL_PATH
args.output = str(MODELS_DIR / args.output)  # тоже автоматически в ./models/

# ───── Валидация и резолв путей ─────
if not args.data.exists():
    raise FileNotFoundError(f"Данные не найдены: {args.data}")

Path(args.output).mkdir(parents=True, exist_ok=True)

print(f"Проект: {PROJECT_ROOT}")
print(f"Модель: {args.model}")
print(f"Данные: {args.data}")
print(f"Сохранение: {args.output}")

try:
    finetune(args)
    print(f"ГОТОВО → {args.output}")
except Exception as e:
    print(f"ОШИБКА: {e}")
    raise