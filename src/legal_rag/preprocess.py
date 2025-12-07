import json
import re
import os
import argparse
import nltk
import pymorphy3
import string
from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Initialize global NLP tools
morph = pymorphy3.MorphAnalyzer()
stemmer = SnowballStemmer("russian")

def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res)

def clean_html(text: str) -> str:
    """Removes HTML tags using BeautifulSoup."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def remove_numbering(text: str) -> str:
    """Removes list numbering (1., 1), a), etc.) and article prefixes."""
    if not text:
        return ""
    
    # Remove article prefixes like "Статья 1. " or "1. " at the start
    text = re.sub(r'^\s*(Статья\s+\d+(\.\d+)*\.?|\d+(\.\d+)*\.)\s*', '', text)
    
    # Remove list items like "1) ", "a) ", "1. " inside text
    text = re.sub(r'(^|\n)\s*\d+[.)]\s+', ' ', text)
    text = re.sub(r'(^|\n)\s*[а-я]\)\s+', ' ', text)
    
    return text

def clean_accordion_artifacts(content: str) -> str:
    """Delete accordeon artifacts and normalize text"""
    if not content:
        return ""
    patterns = [
        r'\n\s*Закрыть\s*\n?',
        r'\n{3,}', # Multiple newlines
        r'\s+',    # Multiple spaces
        r'^\s+|\s+$', # spaces in begin/end 
    ]

    for pattern in patterns:
        content = re.sub(pattern, ' ', content, flags=re.MULTILINE)
    return content.strip()

def remove_emoji(text):
    if not text:
        return ""
    emoji_pattern = re.compile("["
    '\U0001F600-\U0001F64F'  # emoticons
    '\U0001F300-\U0001F5FF'  # symbols & pictographs
    '\U0001F680-\U0001F6FF'  # transport & map symbols
    '\U0001F1E0-\U0001F1FF'  # flags (iOS)
    '\U00002702-\U000027B0'  # Dingbats
    '\U000024C2-\U0001F251'  # Enclosed characters
    '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    '\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    '\U0001F6C0-\U0001F6D0'  # Additional symbols
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_for_nlp(text: str) -> list[str]:
    """
    Strict normalization for NLP tasks:
    - Lowercase
    - Remove punctuation
    - Remove digits
    - Remove stopwords
    Returns a list of tokens.
    """
    if not text:
        return []
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + "«»—–"))
    
    # 3. Remove digits
    text = re.sub(r'\d+', '', text)
    
    # 4. Tokenize
    tokens = nltk.word_tokenize(text, language="russian")
    
    # 5. Remove stopwords
    stop_words = set(stopwords.words('russian'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    return tokens

def lemmatize_text(text: str) -> str:
    """Lemmatizes text using pymorphy3 after strict normalization."""
    tokens = normalize_for_nlp(text)
    lemmas = [morph.parse(word)[0].normal_form for word in tokens]
    return " ".join(lemmas)

def stem_text(text: str) -> str:
    """Stems text using NLTK SnowballStemmer after strict normalization."""
    tokens = normalize_for_nlp(text)
    stems = [stemmer.stem(word) for word in tokens]
    return " ".join(stems)

def clean_text_readable(text):
    """
    Cleans the input text for READABILITY (LLM input).
    Preserves case, punctuation, and sentence structure.
    Removes HTML, numbering, emojis, artifacts.
    """
    if not text:
        return ""

    # 1. Remove HTML
    text = clean_html(text)

    # 2. Remove emojis
    text = remove_emoji(text)
    
    # 3. Remove numbering
    text = remove_numbering(text)

    # 4. Clean artifacts and normalize whitespace
    text = clean_accordion_artifacts(text)
    
    return text

def main():
    parser = argparse.ArgumentParser(description="Preprocess legal articles.")
    parser.add_argument("--input", type=str, default="data/raw/all_articles.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="data/processed/articles_cleaned.json", help="Output JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Download NLTK resources
    download_nltk_resources()

    with open(args.input, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"Preprocessing {len(articles)} articles...")
    processed_articles = []
    
    for article in articles:
        raw_content = article.get('content', '')
        
        # Readable cleaning (for LLMs)
        cleaned_content = clean_text_readable(raw_content)
        cleaned_title = clean_text_readable(article.get('title', ''))
        
        # NLP processing (Strict)
        lemma_content = lemmatize_text(cleaned_content)
        stem_content = stem_text(cleaned_content)
        
        processed_articles.append({
            "url": article['url'],
            "title": cleaned_title,
            "content": cleaned_content,       # Readable
            "lemma_content": lemma_content,   # ML-ready
            "stem_content": stem_content,     # ML-ready
            "raw_content": raw_content        # Backup
        })

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned articles to {args.output}")

if __name__ == "__main__":
    main()
