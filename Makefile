.PHONY: help test test-cov install install-dev lint format clean

help:
	@echo "Доступные команды:"
	@echo "  make install      - Установить базовые зависимости"
	@echo "  make install-dev  - Установить зависимости для разработки"
	@echo "  make test         - Запустить тесты"
	@echo "  make test-cov     - Запустить тесты с покрытием кода"
	@echo "  make lint         - Проверить код линтером"
	@echo "  make format       - Отформатировать код"
	@echo "  make clean        - Очистить временные файлы"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov=scripts --cov-report=html --cov-report=term

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true

