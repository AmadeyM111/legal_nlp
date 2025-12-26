"""
Базовый абстрактный класс для стратегий скрапинга юридических документов.
Использует паттерн Strategy для поддержки различных источников данных.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseStrategy(ABC):
    """
    Абстрактный базовый класс для стратегий скрапинга.
    
    Каждая стратегия должна реализовать:
    - get_article_links: получение списка URL статей
    - parse_article: парсинг HTML контента статьи
    """
    
    @abstractmethod
    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        """
        Получает список URL статей для скрапинга.
        
        Args:
            start_url: Начальный URL для поиска статей
            limit: Максимальное количество статей для получения
            
        Returns:
            Список URL статей
        """
        pass
    
    @abstractmethod
    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Парсит HTML контент статьи и извлекает структурированные данные.
        
        Args:
            url: URL статьи
            html_content: HTML контент страницы
            
        Returns:
            Словарь с ключами:
            - url: URL статьи
            - title: Заголовок статьи
            - content: Текст статьи
        """
        pass

