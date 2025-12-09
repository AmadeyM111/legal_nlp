import json
import argparse
from sklearn.model_selection import train_test_split

def convert_to_alpaca_format(qa_data):
    """
    Converts Q&A data to Alpaca instruction format.
    
    Format:
    {
      "instruction": "Ответь на вопрос клиента как старший юрист РФ.",
      "input": "Question text",
      "output": "Based on [article], the answer is..."
    }
    """
    alpaca_data = []
    
    for item in qa_data:
        question = item.get('question', '').strip()
        context = item.get('context', '').strip()
        article_title = item.get('article_title', '').strip()
        article_url = item.get('article_url', '').strip()
        
        if not question or not context:
            continue
        
        # Create a concise answer using the context
        # For fine-tuning, we want the model to reference the article
        output = f"Согласно {article_title}, {context[:500]}..."
        
        alpaca_item = {
            "instruction": "Ответь на вопрос клиента как старший юрист РФ. Укажи применимую статью закона.",
            "input": question,
            "output": output,
            "metadata": {
                "article_title": article_title,
                "article_url": article_url
            }
        }
        
        alpaca_data.append(alpaca_item)
    
    return alpaca_data

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LoRA fine-tuning.")
    parser.add_argument("--input", type=str, default="data/processed/synthetic_qa_cleaned.json", help="Input cleaned Q&A file")
    parser.add_argument("--output_dir", type=str, default="data/training/", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"Converting {len(qa_data)} Q&A pairs to Alpaca format...")
    alpaca_data = convert_to_alpaca_format(qa_data)
    
    print(f"Converted {len(alpaca_data)} items.")
    
    # Train/test split
    train_data, test_data = train_test_split(
        alpaca_data, 
        test_size=args.test_size, 
        random_state=42
    )
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train.json")
    test_path = os.path.join(args.output_dir, "test.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved test data to {test_path}")
    
    # Show sample
    print("\n--- Sample Training Example ---")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
