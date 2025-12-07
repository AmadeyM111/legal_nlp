import json
import os
import numpy as np
from collections import defaultdict
from urllib.parse import urlparse

def get_code_from_url(url):
    """
    Extracts the legal code abbreviation from the URL.
    Assumes URLs like https://www.zakonrf.info/gk/1/ or similar structures.
    """
    path = urlparse(url).path
    parts = path.strip('/').split('/')
    if parts:
        # Common abbreviations in URLs: gk, tk, nk, jk, uk
        code = parts[0].lower()
        if code in ['gk', 'tk', 'nk', 'jk', 'uk']:
            return code.upper()
    return "OTHER"

def main():
    input_file = "data/processed/synthetic_qa.json"
    
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    print(f"Total Q&A pairs: {total_items}")

    code_counts = defaultdict(int)
    question_lengths = []
    context_lengths = []

    for item in data:
        url = item.get('article_url', '')
        code = get_code_from_url(url)
        code_counts[code] += 1
        
        question = item.get('question', '')
        context = item.get('context', '')
        
        question_lengths.append(len(question))
        context_lengths.append(len(context))

    print("\n--- Distribution by Legal Code ---")
    for code, count in sorted(code_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_items) * 100 if total_items > 0 else 0
        print(f"{code}: {count} ({percentage:.1f}%)")

    print("\n--- Text Length Statistics (characters) ---")
    if question_lengths:
        print("Questions:")
        print(f"  Mean: {np.mean(question_lengths):.1f}")
        print(f"  Median: {np.median(question_lengths):.1f}")
        print(f"  Min: {np.min(question_lengths)}")
        print(f"  Max: {np.max(question_lengths)}")
    
    if context_lengths:
        print("Contexts:")
        print(f"  Mean: {np.mean(context_lengths):.1f}")
        print(f"  Median: {np.median(context_lengths):.1f}")
        print(f"  Min: {np.min(context_lengths)}")
        print(f"  Max: {np.max(context_lengths)}")

if __name__ == "__main__":
    main()
