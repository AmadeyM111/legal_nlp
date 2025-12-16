import json

with open('/Users/antonamadeus/github-projects/Active/experimental/data/processed/enhanced_balanced_legal_dataset.json', encoding='utf-8') as f:
    data = json.load(f)

lengths = [len(item['messages'][-1]['content']) for item in data]  # длина assistant ответа

average = sum(lengths) / len(lengths) if lengths else 0
print(f"Средняя длина: {average:.0f} символов")