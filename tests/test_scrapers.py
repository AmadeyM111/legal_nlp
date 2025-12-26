"""
Unit tests for web scrapers.
"""

import pytest
from pathlib import Path
import sys

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.base import BaseStrategy
from src.scrapers.zakonrf import ZakonRFStrategy
from src.scrapers.sudact import SudactStrategy


class TestBaseStrategy:
    """Тесты для базового класса стратегии."""
    
    def test_base_strategy_is_abstract(self):
        """Проверка, что BaseStrategy является абстрактным классом."""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_base_strategy_has_methods(self):
        """Проверка наличия абстрактных методов."""
        assert hasattr(BaseStrategy, 'get_article_links')
        assert hasattr(BaseStrategy, 'parse_article')


class TestZakonRFStrategy:
    """Тесты для стратегии zakonrf.info."""
    
    def test_strategy_instantiation(self):
        """Проверка создания экземпляра стратегии."""
        strategy = ZakonRFStrategy()
        assert isinstance(strategy, BaseStrategy)
    
    def test_get_article_links(self):
        """Проверка генерации ссылок на статьи."""
        strategy = ZakonRFStrategy()
        start_url = "https://www.zakonrf.info/gk/{}/"
        limit = 5
        
        links = strategy.get_article_links(start_url, limit)
        
        assert isinstance(links, list)
        assert len(links) == limit
        assert all(isinstance(link, str) for link in links)
        assert all("zakonrf.info" in link for link in links)
    
    def test_parse_article_structure(self):
        """Проверка структуры результата парсинга."""
        strategy = ZakonRFStrategy()
        url = "https://www.zakonrf.info/gk/1/"
        html_content = """
        <html>
            <h1 class="law-element__h1">Статья 1. Основные начала</h1>
            <div class="law-element__body">
                <p>Текст статьи...</p>
            </div>
        </html>
        """
        
        result = strategy.parse_article(url, html_content)
        
        assert isinstance(result, dict)
        assert 'url' in result
        assert 'title' in result
        assert 'content' in result
        assert result['url'] == url


class TestSudactStrategy:
    """Тесты для стратегии sudact.ru."""
    
    def test_strategy_instantiation(self):
        """Проверка создания экземпляра стратегии."""
        strategy = SudactStrategy()
        assert isinstance(strategy, BaseStrategy)
    
    def test_parse_article_structure(self):
        """Проверка структуры результата парсинга."""
        strategy = SudactStrategy()
        url = "https://sudact.ru/law/statia-1/"
        html_content = """
        <html>
            <h1>Статья 1. Название</h1>
            <div class="law-document-content">
                <p>Текст статьи...</p>
            </div>
        </html>
        """
        
        result = strategy.parse_article(url, html_content)
        
        assert isinstance(result, dict)
        assert 'url' in result
        assert 'title' in result
        assert 'content' in result
        assert result['url'] == url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

