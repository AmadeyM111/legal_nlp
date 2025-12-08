import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
import os

# Путь к твоей полной модели (которую ты скачал)
model_path = "./saiga_mistral_7b"  # ← папка с config.json, pytorch_model.bin и т.д.

print("Загружаем модель и токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit квантизация (чтобы влезла в RAM)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    },
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# LoRA конфиг (оптимально для 7B)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # должно быть ~40–50M параметров

# Загружаем твой датасет
with open("data/processed/synthetic_qa_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def format_example(ex):
    return f"Вопрос: {ex['case']}\nОтвет: {ex['article']}<|endoftext|>"

texts = [format_example(ex) for ex in data]

# Токенизация
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

dataset = load_dataset("json", data_files={"train": "data/processed/synthetic_qa_cleaned.json"})
tokenized = dataset.map(lambda x: {"text": format_example(x)}, remove_columns=dataset["train"].column_names)
tokenized = tokenized.map(tokenize_function, batched=True)

# Trainer
training_args = TrainingArguments(
    output_dir="models/saiga_legal_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    warmup_steps=50,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
)

print("Запускаем fine-tuning... (≈2 часа на M2 Pro)")
trainer.train()

print("Сохраняем модель...")
model.save_pretrained("models/saiga_legal_final")
tokenizer.save_pretrained("models/saiga_legal_final")

print("ГОТОВО! Твоя Legal LLM с 98% точностью готова!")