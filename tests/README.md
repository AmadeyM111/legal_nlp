# Тесты проекта Legal NLP

## Запуск тестов

### Все тесты
```bash
pytest tests/ -v
```

### Конкретный файл
```bash
pytest tests/test_scrapers.py -v
```

### Конкретный тест
```bash
pytest tests/test_scrapers.py::TestZakonRFStrategy::test_strategy_instantiation -v
```

### С покрытием кода
```bash
pytest tests/ --cov=src --cov=scripts --cov-report=html
```

## Структура тестов

- `test_scrapers.py` - Тесты для web scrapers
- `test_preprocessing.py` - Тесты для пайплайна предобработки данных
- `test_rag_system.py` - Тесты для RAG системы
- `conftest.py` - Конфигурация pytest и общие фикстуры

## Требования

Установите зависимости для разработки:
```bash
pip install -e ".[dev]"
# или
pip install pytest pytest-cov
```

