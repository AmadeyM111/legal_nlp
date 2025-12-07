import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
from .base import BaseStrategy

logger = logging.getLogger(__name__)

class SudactStrategy(BaseStrategy):
    """
    Strategy for scraping sudact.ru.
    """

    def get_article_links(self, start_url: str, limit: int) -> List[str]:
        links = []
        try:
            logger.info(f"Fetching ToC from {start_url}")
            response = requests.get(start_url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False)
            if response.status_code != 200:
                logger.error(f"Failed to fetch ToC: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Sudact ToC usually has links in a list. 
            # We look for links that look like article links.
            # This selector might need adjustment based on actual page structure.
            # Often they are in <ul class="law-chapters"> or similar.
            # Let's try a generic approach: find all links containing 'statia-'
            
            all_links = soup.find_all('a', href=True)
            for a in all_links:
                href = a['href']
                if 'statia-' in href and '/law/' in href:
                    full_url = "https://sudact.ru" + href if href.startswith('/') else href
                    if full_url not in links:
                        links.append(full_url)
                        if len(links) >= limit:
                            break
            
            logger.info(f"Found {len(links)} article links.")
            return links

        except Exception as e:
            logger.error(f"Error fetching ToC: {e}")
            return []

    def parse_article(self, url: str, html_content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Sudact structure
        # Title: h1
        # Content: div class="law-document-content" or similar. 
        # Based on previous scrapes, it might be generic.
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
        
        content = ""
        # Try common content containers
        content_div = soup.find('div', class_='law-document-content') # Common in Sudact
        if not content_div:
             content_div = soup.find('div', class_='doc-content') # Another variant

        if content_div:
            # Preserve paragraphs
            for p in content_div.find_all('p'):
                content += p.get_text(strip=True) + "\n\n"
        else:
            # Fallback: try to find text after h1
            # This is risky, but better than nothing for a universal scraper
            content = soup.get_text(strip=True)

        return {
            "url": url,
            "title": title,
            "content": content.strip()
        }
