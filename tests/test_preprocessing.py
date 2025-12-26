"""
Unit tests for data preprocessing pipeline.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocess_sft_pipeline import DataPreprocessor, ProcessingStats


class TestDataPreprocessor:
    """Тесты для класса предобработки данных."""
    
    def test_preprocessor_initialization(self):
        """Проверка инициализации препроцессора."""
        preprocessor = DataPreprocessor(seed=42)
        assert preprocessor.seed == 42
        assert isinstance(preprocessor.stats, ProcessingStats)
    
    def test_clean_text(self):
        """Проверка очистки текста."""
        preprocessor = DataPreprocessor()
        
        # Тест удаления HTML артефактов
        text = 'id="test">Текст с артефактами'
        cleaned = preprocessor.clean_text(text)
        assert 'id=' not in cleaned
        assert '>' not in cleaned or cleaned.startswith('>')
        
        # Тест нормализации пробелов
        text = "Текст   с    множественными     пробелами"
        cleaned = preprocessor.clean_text(text)
        assert '  ' not in cleaned
    
    def test_extract_article_number(self):
        """Проверка извлечения номера статьи."""
        preprocessor = DataPreprocessor()
        
        # Тест обычного номера
        text = "Статья 1. Название статьи"
        article_num = preprocessor._extract_article_number(text)
        assert article_num == "1"
        
        # Тест составного номера
        text = "Статья 1 2. Название"
        article_num = preprocessor._extract_article_number(text)
        assert article_num == "1_2"
        
        # Тест без номера
        text = "Просто текст без номера"
        article_num = preprocessor._extract_article_number(text)
        assert article_num == "unknown"
    
    def test_normalize_format_case_article(self):
        """Проверка нормализации формата case/article."""
        preprocessor = DataPreprocessor()
        
        records = [{
            'source_id': 'test_1',
            'group_id': 'gk_1',
            'code': 'gk',
            'article_number': '1',
            'raw_case': 'Вопрос о договорах',
            'raw_article': 'Договором признается соглашение...',
            'is_messages_format': False
        }]
        
        normalized = preprocessor.normalize_format(records)
        
        assert len(normalized) == 1
        assert 'messages' in normalized[0]
        assert len(normalized[0]['messages']) == 3
        assert normalized[0]['messages'][0]['role'] == 'system'
        assert normalized[0]['messages'][1]['role'] == 'user'
        assert normalized[0]['messages'][2]['role'] == 'assistant'
    
    def test_normalize_format_messages(self):
        """Проверка нормализации формата messages."""
        preprocessor = DataPreprocessor()
        
        records = [{
            'source_id': 'test_1',
            'group_id': 'gk_1',
            'code': 'gk',
            'article_number': '1',
            'raw_case': 'Вопрос о договорах',
            'raw_article': 'Договором признается соглашение...',
            'is_messages_format': True
        }]
        
        normalized = preprocessor.normalize_format(records)
        
        assert len(normalized) == 1
        assert 'messages' in normalized[0]
        assert len(normalized[0]['messages']) == 3
    
    def test_quality_filters(self):
        """Проверка фильтров качества."""
        preprocessor = DataPreprocessor()
        
        # Валидная запись
        valid_record = {
            'messages': [
                {'role': 'system', 'content': 'System prompt'},
                {'role': 'user', 'content': 'Достаточно длинный вопрос пользователя'},
                {'role': 'assistant', 'content': 'Достаточно длинный ответ ассистента с информацией'}
            ]
        }
        
        # Слишком короткий user
        short_user_record = {
            'messages': [
                {'role': 'user', 'content': 'К'},
                {'role': 'assistant', 'content': 'Длинный ответ'}
            ]
        }
        
        # Слишком короткий assistant
        short_assistant_record = {
            'messages': [
                {'role': 'user', 'content': 'Длинный вопрос'},
                {'role': 'assistant', 'content': 'К'}
            ]
        }
        
        records = [
            {'messages': valid_record['messages']},
            {'messages': short_user_record['messages']},
            {'messages': short_assistant_record['messages']}
        ]
        
        filtered = preprocessor.quality_filters(records)
        
        # Должна остаться только валидная запись
        assert len(filtered) == 1
        assert filtered[0]['messages'] == valid_record['messages']
    
    def test_deduplication(self):
        """Проверка дедупликации."""
        preprocessor = DataPreprocessor()
        
        # Две одинаковые записи
        duplicate = {
            'messages': [
                {'role': 'user', 'content': 'Вопрос'},
                {'role': 'assistant', 'content': 'Ответ'}
            ]
        }
        
        records = [
            {'messages': duplicate['messages']},
            {'messages': duplicate['messages']},
            {
                'messages': [
                    {'role': 'user', 'content': 'Другой вопрос'},
                    {'role': 'assistant', 'content': 'Другой ответ'}
                ]
            }
        ]
        
        deduplicated = preprocessor.deduplication(records)
        
        # Должно остаться 2 уникальные записи
        assert len(deduplicated) == 2
    
    def test_split_train_val_test(self):
        """Проверка разделения на train/val/test."""
        preprocessor = DataPreprocessor(seed=42)
        
        # Создаем записи с разными group_id
        records = []
        for i in range(10):
            records.append({
                'group_id': f'gk_{i}',
                'code': 'gk',
                'messages': [
                    {'role': 'user', 'content': f'Вопрос {i}'},
                    {'role': 'assistant', 'content': f'Ответ {i}'}
                ]
            })
        
        train, val, test = preprocessor.split_train_val_test(
            records,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # Проверка пропорций (примерно)
        assert len(train) >= 7
        assert len(val) >= 1
        assert len(test) >= 1
        assert len(train) + len(val) + len(test) == len(records)
        
        # Проверка отсутствия пересечений
        train_groups = {r['group_id'] for r in train}
        val_groups = {r['group_id'] for r in val}
        test_groups = {r['group_id'] for r in test}
        
        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0


class TestProcessingStats:
    """Тесты для статистики обработки."""
    
    def test_stats_initialization(self):
        """Проверка инициализации статистики."""
        stats = ProcessingStats()
        assert stats.initial_count == 0
        assert stats.removed_reasons is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

