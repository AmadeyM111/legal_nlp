import json
import re
import argparse
import pandas as pd

def normalize_text(text):
    """Normalizes whitespace."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def has_valid_content(text):
    """
    Checks if the question has a verb or a question word.
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Verbs and modal words
    verbs = [
        'должен', 'может', 'обязан', 'имеет', 'нужно', 'надо', 'возможно', 'требуется',
        'является', 'существует', 'грозит', 'делать', 'поступить', 'считается',
        'применяется', 'регулирует', 'определяет', 'вправе'
    ]
    
    # Question words
    question_words = [
        'какой', 'какая', 'какие', 'какое',
        'кто', 'что', 'где', 'когда', 'почему', 'зачем', 'сколько',
        'как', 'чей', 'чья', 'чье', 'чьи',
        'могу', 'может', 'ли'
    ]
    
    has_verb = any(v in text_lower for v in verbs)
    has_q_word = any(qw in text_lower for qw in question_words)
    
    return has_verb or has_q_word

def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate synthetic Q&A dataset.")
    parser.add_argument("--input", type=str, default="data/processed/synthetic_qa.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="data/processed/synthetic_qa_cleaned.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    df = pd.DataFrame(data)
    initial_count = len(df)
    print(f"Initial count: {initial_count}")

    # 1. Normalize whitespace in questions
    df['question'] = df['question'].apply(normalize_text)

    # 2. Deduplicate by question text
    df = df.drop_duplicates(subset=['question'])
    dedup_count = len(df)
    print(f"After deduplication: {dedup_count} (removed {initial_count - dedup_count})")

    # 3. Filter by length (< 20 chars)
    # "Что это?" is too short for a legal context usually
    df = df[df['question'].str.len() >= 20]
    length_filter_count = len(df)
    print(f"After length filter (>= 20 chars): {length_filter_count} (removed {dedup_count - length_filter_count})")

    # 4. Heuristic filter (Verbs / Question words)
    df = df[df['question'].apply(has_valid_content)]
    final_count = len(df)
    print(f"After heuristic filter: {final_count} (removed {length_filter_count - final_count})")

    # Save result
    result_data = df.to_dict(orient='records')
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset to {args.output}")
    
    # Show sample of kept questions
    print("\n--- Sample of cleaned questions ---")
    print(df['question'].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
