import json

# Load the file with data
with open("all_articles.json", "r", encoding="utf-8") as f:
    article = json.load(f)

print(f"First article: {article[0]['title']}")
print(f"Last article: {article[-1]['title']}")
print(f"Total articles: {len(article)}")