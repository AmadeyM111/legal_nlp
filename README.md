# Дообучение и промптинг LLM для предсказания применимой нормы закона

## Цель  
Создать модель, которая по описанию ситуации клиента определяет соответствующую статью российского законодательства.

## Структура датасета  
```json
{
  "article_url": "https://www.zakonrf.info/gk/2/",
  "article_title": "ГК РФ. Отношения, регулируемые гражданским законодательством (действующая редакция)",
  "question": "Могут ли государственные органы выступать участниками гражданских правоотношений?",
  "context": "Гражданское законодательство определяет правовое положение участников..."
}
```

- Источник данных: https://www.zakonrf.info  
- Кейсы сгенерированы Gemini Pro 3  
- Объём: 2270 примеров  
  - train: 1816  
  - test: 454  

## Шаги выполнения

### 1. Подготовка данных  
Собрано 2270 примеров в JSON-формате.  
Разделение: 80 % train / 20 % test.

### 2. Baseline (Prompt-only)  
Реализован few-shot prompting (3–5 примеров в промпте).  
Инструкция: «Ты — юридический ассистент. Определи статью закона, регулирующую описанную ситуацию».

### 3. Fine-tuning / LoRA  
- Базовая модель: `saiga_mistral_7b_gguf`  
- Дообучение через LoRA (PEFT)

**Параметры LoRA:**
```python
lora_config = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**Параметры обучения:**
```python
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 50,
    "max_steps": 1000,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "output_dir": "models/legal-saiga-lora",
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
    "report_to": "none"
}
```

**Потребляемые ресурсы:**
- RAM: 7–11 ГБ  
- Диск: ~8 ГБ  
- Время обучения: 40–90 минут  
- Время инференса: ~0.5 сек на запрос