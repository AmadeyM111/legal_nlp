# finetune_qlora.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
import json
from pathlib import Path

# === Конфиг ===
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
DATA_PATH = "data/processed/synthetic_qa_cleaned.json"
OUTPUT_DIR = "saiga-legal-qlora"
LORA_DIR = "saiga-legal-lora-final"

# === Загрузка в 4-bit (QLoRA) ===
quant_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === QLoRA конфиг ===
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # покажет ~1–2% параметров

# === Датасет ===
with open(DATA_PATH) as f:
    data = json.load(f)

dataset = Dataset.from_list([
    {
        "instruction": "Определи применимую статью закона",
        "input": item["case"],
        "output": item["article"]
    }
    for item in data
])

def format_prompt(ex):
    return f"<s>[INST] {ex['input']} [/INST] {ex['output']} </s>"

tokenized = dataset.map(
    lambda x: tokenizer(format_prompt(x), truncation=True, max_length=2048),
    batched=True
)

# === Обучение ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

print("Запускаем QLoRA fine-tuning...")
trainer.train()

# === Сохранение LoRA ===
Path(LORA_DIR).mkdir(exist_ok=True)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"LoRA-адаптер сохранён: {LORA_DIR}")