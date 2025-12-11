#!/usr/bin/env python3
import json
import argparse
import logging
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

# Set up logging first before any other processing
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "synthetic_qa_cleaned.json")
parser.add_argument("--model", type=str, default="saiga_mistral_7b_merged",   # ← теперь просто имя папки!
                    help="Либо имя папки в ./models/, либо полное HF repo (IlyaGusev/...)")
parser.add_argument("--output", type=str, default="models/saiga-legal-7b")
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--log_file", type=str, default="fine_tuning.log", help="Path to log file")
args = parser.parse_args()

# Setup logging first thing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler()  # Also log to console
    ]
)

logger = logging.getLogger(__name__)

# Now we can safely use logger
logger.info("Setting up fine-tuning script...")

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
logger.info(f"Backend detected: {BACKEND}")

def finetune(args):
    if BACKEND == "mlx":
        # Apple Silicon
        from mlx_lm import load, lora
        logger.info("Loading model with MLX backend")
        
        model, tokenizer = load(args.model)
        
        logger.info(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Validate data format
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
        
        train_data = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {i}")
                continue
                
            user_content = item.get("case") or item.get("question", "")
            assistant_content = item.get("article") or item.get("answer", "") or item.get("context", "")
            
            if not user_content.strip() or not assistant_content.strip():
                logger.warning(f"Skipping item {i} with empty content")
                continue
                
            train_data.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])
        
        if len(train_data) == 0:
            raise ValueError("No valid training data found after filtering")
            
        logger.info(f"Training with {len(train_data)} examples")
        
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
        logger.info(f"Model saved to {args.output}")
    else:
        # GPU/CPU backend - Transformers + PEFT
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

        # Try to use Unsloth if available
        try:
            from unsloth import FastLanguageModel
            USE_UNSLOTH = True
            logger.info("Unsloth detected → до 2.5× быстрее + 70% меньше VRAM")
        except ImportError:
            USE_UNSLOTH = False
            logger.info("Transformers + QLoRA (Unsloth not available)")

        logger.info(f"Loading model: {args.model}")
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
            
            # Полностью исключаем bitsandbytes из процесса - используем прямую загрузку с 4-bit квантованием
            try:
                # Проверяем, есть ли файлы индекса для шардинга
                import os
                index_file = os.path.join(args.model, "pytorch_model.bin.index.json")
                safetensors_index = os.path.join(args.model, "model.safetensors.index.json")
                
                if os.path.exists(index_file) or os.path.exists(safetensors_index):
                    # Модель шардирована - используем временную папку для offload
                    import tempfile
                    with tempfile.TemporaryDirectory() as offload_folder:
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model,
                            load_in_4bit=True,
                            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                            device_map="auto",
                            offload_folder=offload_folder,
                        )
                else:
                    # Модель не шардирована - загружаем напрямую
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        load_in_4bit=True,
                        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                        device_map="auto",
                    )
            except Exception as e:
                logger.warning(f"4-bit loading failed ({e}), falling back to normal loading")
                # Если 4-bit квантование не работает, пробуем без него
                try:
                    import os
                    index_file = os.path.join(args.model, "pytorch_model.bin.index.json")
                    safetensors_index = os.path.join(args.model, "model.safetensors.index.json")
                    
                    if os.path.exists(index_file) or os.path.exists(safetensors_index):
                        # Модель шардирована - используем временную папку для offload
                        import tempfile
                        with tempfile.TemporaryDirectory() as offload_folder:
                            model = AutoModelForCausalLM.from_pretrained(
                                args.model,
                                torch_dtype=torch.float16,  # используем float16 вместо bfloat16
                                device_map="auto",
                                offload_folder=offload_folder,
                            )
                    else:
                        # Модель не шардирована - загружаем напрямую
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model,
                            torch_dtype=torch.float16,  # используем float16 вместо bfloat16
                            device_map="auto",
                        )
                except Exception as e2:
                    logger.warning(f"Normal loading also failed ({e2}), trying without device_map")
                    # Если ничего не работает, пробуем загрузить без device_map
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    # Перемещаем модель на GPU вручную, если доступно
                    if torch.cuda.is_available():
                        model = model.to('cuda')
            
            # Обработка LoRA с обработкой ошибок нехватки памяти
            try:
                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, LoraConfig(
                    r=args.rank,
                    lora_alpha=32,
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                ))
            except RuntimeError as e:
                if "MPS backend out of memory" in str(e) or "out of memory" in str(e).lower():
                    logger.warning("MPS out of memory during prepare_model_for_kbit_training, trying alternative approach")
                    # Удаляем модель из памяти и пробуем с меньшим потреблением памяти
                    import gc
                    del model
                    gc.collect()
                    if torch.mps.is_available():
                        torch.mps.empty_cache()
                    
                    # Загружаем модель заново с меньшим потреблением памяти
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.float16,
                        device_map="mps",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                    
                    # Применяем LoRA без prepare_model_for_kbit_training
                    model = get_peft_model(model, LoraConfig(
                        r=args.rank,
                        lora_alpha=32,
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                        lora_dropout=0.05,
                        task_type="CAUSAL_LM",
                    ))
                    logger.info("Successfully applied LoRA with memory-optimized approach")
                else:
                    raise e

        # Данные
        logger.info(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Validate data format
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
            
        # Filter out invalid entries
        filtered_raw = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {i}")
                continue
                
            user_content = item.get("case") or item.get("question", "") or item.get("article_title", "")
            ans_content = item.get("article") or item.get("answer", "") or item.get("context", "")
            
            if not user_content.strip() or not ans_content.strip():
                logger.warning(f"Skipping item {i} with empty content")
                continue
                
            filtered_raw.append(item)
            
        if len(filtered_raw) == 0:
            raise ValueError("No valid training data found after filtering")
            
        logger.info(f"Processing {len(filtered_raw)} valid examples")
        # Check the first few examples to understand the data format
        if filtered_raw:
            sample_item = filtered_raw[0]
            logger.info(f"Sample data keys: {list(sample_item.keys())}")
            user_sample = sample_item.get("case") or sample_item.get("question", "") or sample_item.get("article_title", "")
            ans_sample = sample_item.get("article") or sample_item.get("answer", "") or sample_item.get("context", "")
            logger.info(f"Sample input (first 100 chars): {user_sample[:100]}...")
            logger.info(f"Sample output (first 100 chars): {ans_sample[:100]}...")
        
        dataset = Dataset.from_list(filtered_raw)
        def formatting_func(ex):
            user = ex.get("case") or ex.get("question", "") or ex.get("article_title", "")
            ans  = ex.get("article") or ex.get("answer", "") or ex.get("context", "")
            text = f"<s>[INST] {user} [/INST] {ans}</s>"
            return {"text": text}
        dataset = dataset.map(lambda x: {"text": formatting_func(x)["text"]})

        # Try to use SFTTrainer if available, otherwise use regular Trainer
        try:
            from trl import SFTTrainer
            # Пытаемся использовать SFTTrainer с разными комбинациями параметров
            try:
                # Пробуем с dataset_text_field и max_seq_length
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    args=TrainingArguments(
                        per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                        gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                        num_train_epochs=3,
                        learning_rate=args.lr,
                        fp16=not is_bfloat16_supported(),
                        bf16=is_bfloat16_supported(),
                        logging_steps=10,
                        save_steps=500,
                        output_dir=args.output,
                        optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                        report_to="none",
                        remove_unused_columns=False,  # Important for custom formatting
                        dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                        dataloader_num_workers=0, # Используем 0 воркеров для экономии памяти
                    ),
                    dataset_text_field="text",
                    max_seq_length=2048,
                )
                logger.info("Using SFTTrainer with dataset_text_field and max_seq_length")
            except TypeError as e:
                if "max_seq_length" in str(e):
                    # Если max_seq_length не поддерживается, пробуем без него
                    try:
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=dataset,
                            args=TrainingArguments(
                                per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                                gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                                num_train_epochs=3,
                                learning_rate=args.lr,
                                fp16=not is_bfloat16_supported(),
                                bf16=is_bfloat16_supported(),
                                logging_steps=10,
                                save_steps=500,
                                output_dir=args.output,
                                optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                                report_to="none",
                                remove_unused_columns=False,  # Important for custom formatting
                                dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                                dataloader_num_workers=0,  # Используем 0 воркеров для экономии памяти
                            ),
                            dataset_text_field="text",
                        )
                        logger.info("Using SFTTrainer with dataset_text_field only")
                    except TypeError:
                        # Если dataset_text_field не поддерживается, используем formatting_func
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=dataset,
                            args=TrainingArguments(
                                per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                                gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                                num_train_epochs=3,
                                learning_rate=args.lr,
                                fp16=not is_bfloat16_supported(),
                                bf16=is_bfloat16_supported(),
                                logging_steps=10,
                                save_steps=500,
                                output_dir=args.output,
                                optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                                report_to="none",
                                remove_unused_columns=False,  # Important for custom formatting
                                dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                                dataloader_num_workers=0,  # Используем 0 воркеров для экономии памяти
                            ),
                            formatting_func=lambda x: x["text"],
                        )
                        logger.info("Using SFTTrainer with formatting_func")
                elif "dataset_text_field" in str(e):
                    # Если dataset_text_field не поддерживается, используем formatting_func
                    trainer = SFTTrainer(
                        model=model,
                        train_dataset=dataset,
                        args=TrainingArguments(
                            per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                            gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                            num_train_epochs=3,
                            learning_rate=args.lr,
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            logging_steps=10,
                            save_steps=500,
                            output_dir=args.output,
                            optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                            report_to="none",
                            remove_unused_columns=False,  # Important for custom formatting
                            dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                            dataloader_num_workers=0,  # Используем 0 воркеров для экономии памяти
                        ),
                        formatting_func=lambda x: x["text"],
                    )
                    logger.info("Using SFTTrainer with formatting_func")
                else:
                    # Если другая ошибка, пробуем без max_seq_length
                    trainer = SFTTrainer(
                        model=model,
                        train_dataset=dataset,
                        args=TrainingArguments(
                            per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                            gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                            num_train_epochs=3,
                            learning_rate=args.lr,
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            logging_steps=10,
                            save_steps=500,
                            output_dir=args.output,
                            optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                            report_to="none",
                            remove_unused_columns=False,  # Important for custom formatting
                            dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                            dataloader_num_workers=0,  # Используем 0 воркеров для экономии памяти
                        ),
                        formatting_func=lambda x: x["text"],
                    )
                    logger.info("Using SFTTrainer with formatting_func after error")
        except ImportError:
            # If SFTTrainer is not available, use regular Trainer with a data collator
            logger.info("SFTTrainer not available, using regular Trainer")
            trainer = Trainer(
                model=model,
                train_dataset=dataset,
                args=TrainingArguments(
                    per_device_train_batch_size=max(1, args.batch // 2),  # Уменьшаем размер батча вдвое для экономии памяти
                    gradient_accumulation_steps=max(1, 16//args.batch),  # Увеличиваем накопление градиентов для компенсации
                    num_train_epochs=3,
                    learning_rate=args.lr,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=10,
                    save_steps=500,
                    output_dir=args.output,
                    optim="adamw_torch",  # Используем torch adamw вместо 8-bit adamw
                    report_to="none",
                    remove_unused_columns=False,  # Important for custom formatting
                    dataloader_pin_memory=False,  # Отключаем закрепление памяти для экономии
                    dataloader_num_workers=0,  # Используем 0 воркеров для экономии памяти
                ),
            )
        
        # Освобождаем память перед началом тренировки
        import gc
        if torch.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        trainer.train()
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        logger.info(f"Model saved to {args.output}")

# ───── Валидация и резолв путей ─────
MODEL_PATH = resolve_model_path(args.model)
args.model = MODEL_PATH
args.output = str(MODELS_DIR / args.output) # тоже автоматически в ./models/

if not args.data.exists():
    raise FileNotFoundError(f"Данные не найдены: {args.data}")

Path(args.output).mkdir(parents=True, exist_ok=True)

logger.info(f"Проект: {PROJECT_ROOT}")
logger.info(f"Модель: {args.model}")
logger.info(f"Данные: {args.data}")
logger.info(f"Сохранение: {args.output}")

try:
    logger.info(f"Starting fine-tuning with model: {args.model}, data: {args.data}")
    finetune(args)
    logger.info(f"Fine-tuning completed successfully → {args.output}")
    print(f"ГОТОВО → {args.output}")
except Exception as e:
    logger.error(f"Error during fine-tuning: {str(e)}")
    print(f"ОШИБКА: {e}")
    raise