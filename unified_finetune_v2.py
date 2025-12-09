#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from datasets import Dataset

# Автоопределение бэкенда и установка нужного
def install_backend():
    try:
        import mlx
        return "mlx"
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
    return "cuda"  # fallback

BACKEND = install_backend()

if BACKEND == "mlx":
    # Apple Silicon — самый быстрый способ
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
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
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
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            )
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

# === Запуск ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/synthetic_qa_cleaned.json", help="Path to training data JSON file")
    parser.add_argument("--model", default="saiga_mistral_7b_merged", help="Base model to fine-tune")
    parser.add_argument("--output", default="models/saiga-legal-7b", help="Output directory for fine-tuned model")
    parser.add_argument("--iters", type=int, default=2000, help="Number of training iterations (MLX only)")  # только для MLX
    parser.add_argument("--batch", type=int, default=4, help="Training batch size")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank parameter")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Training data file not found: {args.data}")
    
    # Validate model exists
    if not Path(args.model).exists() and not args.model.startswith("http"):
        # Try to load from hub to check if it exists
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(args.model)
        except:
            raise ValueError(f"Model not found: {args.model}")

    try:
        finetune(args)
        print(f"Готово → {args.output}")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise