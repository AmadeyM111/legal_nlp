"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Возвращает путь к корню проекта."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Возвращает путь к директории с тестовыми данными."""
    test_data = project_root_path / "tests" / "test_data"
    test_data.mkdir(exist_ok=True)
    return test_data

