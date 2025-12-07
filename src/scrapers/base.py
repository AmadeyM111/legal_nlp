from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseStrategy(ABC):
    """
    Abstract base class for scraping strategies.
    """

    @abstractmethod
    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        """
        Extracts article links from the start URL (Table of Contents).
        """
        pass

    @abstractmethod
    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Parses a single article page.
        """
        pass
