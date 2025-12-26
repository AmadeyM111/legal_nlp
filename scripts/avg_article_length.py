import json
import logging
import argparse
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Вычисляет среднюю длину статей в датасете")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/enhanced_balanced_legal_dataset.json",
        help="Путь к JSON файлу с датасетом"
    )
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Файл не найден: {dataset_path}")
        return
    
    try:
        with open(dataset_path, encoding='utf-8') as f:
            data = json.load(f)
        
        lengths = [len(item['messages'][-1]['content']) for item in data if 'messages' in item and len(item['messages']) > 0]
        
        if not lengths:
            logger.warning("Не найдено ни одной записи с messages")
            return
        
        average = sum(lengths) / len(lengths)
        logger.info(f"Средняя длина assistant ответа: {average:.0f} символов")
        logger.info(f"Минимальная длина: {min(lengths)} символов")
        logger.info(f"Максимальная длина: {max(lengths)} символов")
        logger.info(f"Всего записей: {len(lengths)}")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")

if __name__ == "__main__":
    main()