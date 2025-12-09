import hashlib
import logging
from pathlib import Path

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compute_hashes.log', encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "models" / "saiga_mistral_7b_merged"
MODEL_ID = "IlyaGusev/saiga_mistral_7b_merged"
OUT_PATH = MODEL_DIR / "saiga_sha256.txt"

def file_sha256(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not MODEL_DIR.exists():
        logging.error(f"Модельная папка не найдена: {MODEL_DIR}")
        raise SystemExit(f"Модельная папка не найдена: {MODEL_DIR}")

    # Find pytorch_model-*.bin files
    target_files = sorted(MODEL_DIR.glob("pytorch_model-*.bin"))
    
    # Add specific files to the list
    additional_files = ["config.json", "tokenizer.model", "pytorch_model.bin.index.json"]
    for filename in additional_files:
        file_path = MODEL_DIR / filename
        if file_path.exists():
            target_files.append(file_path)
        else:
            logging.warning(f"Файл не найден: {file_path}")

    if not target_files:
        logging.error("Не найдены файлы для вычисления хешей")
        raise SystemExit("Не найдены файлы для вычисления хешей")

    lines = []
    for f in target_files:
        logging.info(f"Считаем SHA256 для {f.name} ...")
        h = file_sha256(f)
        line = f"{h}  {f.name}"
        logging.info(line)
        lines.append(line)

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"\n✅ Хеши сохранены в {OUT_PATH}")

if __name__ == "__main__":
    main()
