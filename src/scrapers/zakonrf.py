import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
import time
import random
from .base import BaseStrategy

logger = logging.getLogger(__name__)

class ZakonRFStrategy(BaseStrategy):
    """
    Strategy for scraping zakonrf.info.
    """

    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        # ZakonRF structure is predictable: /{code}/{article_num}/
        # start_url in config will be treated as the base template e.g. "https://www.zakonrf.info/{code}/{}/"
        # We generate links instead of crawling ToC for simplicity and reliability as seen in previous steps.
        
        links = []
        # Extract code from start_url or pass it? 
        # The start_url in sources.json for zakonrf should be the template.
        
        for i in range(1, limit + 1):
            link = start_url.format(i)
            links.append(link)
            
        return links

    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title_tag = soup.select_one('h1.law-element__h1')
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

        content_div = soup.select_one('div.law-element__body')
        content = ""
        if content_div:
            for p in content_div.find_all('p'):
                content += p.get_text(strip=True) + "\n\n"
        
        return {
            "url": url,
            "title": title,
            "content": content.strip()
        }
