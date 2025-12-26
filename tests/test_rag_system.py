"""
Unit tests for RAG system.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.legal_rag.rag_system import build_vector_db, search_legal_context


class TestRAGSystem:
    """Тесты для RAG системы."""
    
    @pytest.fixture
    def sample_articles(self):
        """Фикстура с примерами статей."""
        return [
            {
                "url": "https://www.zakonrf.info/gk/420/",
                "title": "ГК РФ. Статья 420. Понятие договора",
                "content": "Договором признается соглашение двух или нескольких лиц об установлении, изменении или прекращении гражданских прав и обязанностей."
            },
            {
                "url": "https://www.zakonrf.info/tk/56/",
                "title": "ТК РФ. Статья 56. Трудовой договор",
                "content": "Трудовой договор - соглашение между работодателем и работником, в соответствии с которым работодатель обязуется предоставить работнику работу по обусловленной трудовой функции."
            }
        ]
    
    @pytest.fixture
    def temp_db_path(self):
        """Фикстура с временным путем к БД."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_db"
    
    def test_build_vector_db_structure(self, sample_articles, temp_db_path):
        """Проверка структуры создания векторной БД."""
        # Создаем временный файл со статьями
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_articles, f, ensure_ascii=False)
            articles_path = f.name
        
        try:
            # Создаем БД
            client, collection = build_vector_db(
                articles_path=articles_path,
                db_path=str(temp_db_path)
            )
            
            # Проверяем, что коллекция создана
            assert collection is not None
            assert collection.name == "russian_law"
            
            # Проверяем количество документов
            count = collection.count()
            assert count > 0
            
        finally:
            # Удаляем временный файл
            Path(articles_path).unlink()
    
    def test_search_legal_context(self, sample_articles, temp_db_path):
        """Проверка поиска в RAG системе."""
        # Создаем временный файл со статьями
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_articles, f, ensure_ascii=False)
            articles_path = f.name
        
        try:
            # Создаем БД
            build_vector_db(
                articles_path=articles_path,
                db_path=str(temp_db_path)
            )
            
            # Выполняем поиск
            results = search_legal_context(
                query="Что такое договор?",
                db_path=str(temp_db_path),
                top_k=2
            )
            
            # Проверяем результаты
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Проверяем структуру результата
            for result in results:
                assert 'text' in result
                assert 'article_title' in result
                assert 'article_url' in result
                assert 'legal_code' in result
                assert isinstance(result['text'], str)
                assert isinstance(result['article_title'], str)
                
        finally:
            # Удаляем временный файл
            Path(articles_path).unlink()
    
    def test_search_empty_query(self, sample_articles, temp_db_path):
        """Проверка обработки пустого запроса."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_articles, f, ensure_ascii=False)
            articles_path = f.name
        
        try:
            build_vector_db(
                articles_path=articles_path,
                db_path=str(temp_db_path)
            )
            
            # Пустой запрос должен вернуть пустой список или обработаться корректно
            results = search_legal_context(
                query="",
                db_path=str(temp_db_path),
                top_k=5
            )
            
            assert isinstance(results, list)
            
        finally:
            Path(articles_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

