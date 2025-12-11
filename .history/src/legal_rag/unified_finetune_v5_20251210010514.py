#!/usr/bin/env python3
import json
import argparse
import logging
from pathlib import Path
from datasets import Dataset

# ──────────────── PROJECT ROOT & SMART MODEL RESOLUTION ────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # src/legal_rag/ → корень проекта
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def resolve_model_path(model_arg: str) -> str:
    """Умно находит модель: локальная папка в ./models/, HF repo или полный путь"""
    # Если это явно путь — возвращаем как есть
    if Path(model_arg).exists():
        return str(Path(model_arg).resolve())

    # Если выглядит как HF репозиторий (user/repo)
    if "/" in model_arg and model_arg.count("/") == 1:
        return model_arg

    # Иначе — ищем в ./models/ по имени папки
    candidate = MODELS_DIR / model_arg
    if candidate.is_dir() and (candidate / "config.json").exists():
        return str(candidate.resolve())

    raise FileNotFoundError(
        f"\nМодель не найдена: {model_arg}\n"
        f"\nПроверено: {candidate}\n"
        f"Доступные локальные модели в {MODELS_DIR}:\n" +
        "\n".join([p.name for p in MODELS_DIR.iterdir()
                    if p.is_dir() and (p/"config.json").exists()][:10])
    )

# ──────────────────────── LOGGING & ARGS ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Универсальный LoRA файн-тюнинг Saiga Mistral 7B")
parser.add_argument("--data", type=Path, default=DATA_DIR / "synthetic_qa_cleaned.json")
parser.add_argument("--model", type=str, default="saiga_mistral_7b_merged", help="Имя папки в ./models/ или HF repo")
parser.add_argument("--output", type=str, default="saiga-legal-7b", help="Имя папки в ./models/ для результата")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--iters", type=int, default=3000, help="Только для Apple Silicon")
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

# ──────────────────────── ВАВТОДЕТЕКТ БЭКЕНДА────────────────────────
def get_backend() -> str:
    try:
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            import mlx
            return "mlx"
    except:
        pass
    return "unsupported"

BACKEND = get_backend()

if BACKEND != "mlx":
    raise RuntimeError(
        "Этот скрипт оптимизирован ТОЛЬКО под Apple Silicon (MLX).\n"
        "На Mac M1/M2/M3/M4 используй mlx-lm — он в 10× быстрее и стабильнее.\n"
        "PyTorch-бэкенд отключён во избежание ошибок и мучений."
    )

logger.info("Apple Silicon обнаружен → используем MLX-LM (самый быстрый способ 2025)")

# ──────────────────────── РЕЗОЛВ ПУТЕЙ ────────────────────────
if not args.data.exists():
    raise FileNotFoundError(f"Файл с данными не найден: {args.data}")

MODEL_PATH = resolve_model_path(args.model)
OUTPUT_PATH = MODELS_DIR / args.output
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger.info(f"Проект: {PROJECT_ROOT}")
logger.info(f"Модель: {MODEL_PATH}")
logger.info(f"Данные: {args.data}")
logger.info(f"Сохранение: {OUTPUT_PATH}")

# ──────────────────────── FINETUNE (ТОЛЬКО MLX) ────────────────────────
def finetune():
    from mlx_lm import load, lora

    logger.info("Загружаем модель...")
    model, tokenizer = load(MODEL_PATH)

    logger.info(f"Загружаем данные из {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("JSON должен быть массивом объектов")

    train_data = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        user = item.get("case") or item.get("question", "") or item.get("article_title", "")
        assistant = item.get("article") or item.get("answer", "") or item.get("context", "")
        if user.strip() and assistant.strip():
            train_data.append([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])

    if not train_data:
        raise ValueError("Ни одного валидного примера не найдено")

    logger.info(f"Подготовлено {len(train_data)} примеров → начинаем обучение")

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


    logger.info(f"Сохраняем в {OUTPUT_PATH}...")
    model.save(str(OUTPUT_PATH))
    tokenizer.save(str(OUTPUT_PATH))
    logger.info("ГОТОВО! Модель сохранена")

# ──────────────────────── ЗАПУСК ────────────────────────
if __name__ == "main__":
    try:
        finetune()
        print(f"\nУСПЕШНО → {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"ОШИБКА: {e}", exc_info=True)
        raise