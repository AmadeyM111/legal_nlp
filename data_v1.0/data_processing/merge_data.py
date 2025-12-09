import json
import os
import re

def merge_data():
    input_file = "data/raw/articles.json"
    output_dir = "data/raw"
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    # Load new scraped data
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            new_articles = json.load(f)
        print(f"Loaded {len(new_articles)} new articles from {input_file}")
    else:
        new_articles = []

    # Load existing combined data
    all_articles_path = os.path.join(output_dir, "all_articles.json")
    existing_articles = []
    if os.path.exists(all_articles_path):
        with open(all_articles_path, "r", encoding="utf-8") as f:
            existing_articles = json.load(f)
        print(f"Loaded {len(existing_articles)} existing articles from {all_articles_path}")

    # Merge: Use dictionary by URL to deduplicate/update
    article_map = {a["url"]: a for a in existing_articles}
    
    # Update with new articles
    for a in new_articles:
        article_map[a["url"]] = a
        
    articles = list(article_map.values())
    print(f"Total unique articles after merge: {len(articles)}")

    
    # Containers for each code
    codes = {
        "GK": [],
        "TK": [],
        "NK": [],
        "JK": []
    }
    
    # Regex to identify code from URL
    # URLs are like https://www.zakonrf.info/gk/1/
    
    for article in articles:
        url = article.get("url", "")
        if "/gk/" in url:
            codes["GK"].append(article)
        elif "/tk/" in url:
            codes["TK"].append(article)
        elif "/nk/" in url:
            codes["NK"].append(article)
        elif "/jk/" in url:
            codes["JK"].append(article)
        else:
            print(f"Unknown code for URL: {url}")
            
    # Save individual files
    for code, data in codes.items():
        filename = f"{code.lower()}_rf_articles.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} articles to {filename}")
        
    # Save all combined (excluding UK for now as per target, but let's include if we want a master list)
    # The user asked for 125 of each of these 4.
    # We can also load UK if it exists and add it to all_articles.
    
    uk_path = os.path.join(output_dir, "uk_rf_articles.json")
    if os.path.exists(uk_path):
        with open(uk_path, "r", encoding="utf-8") as f:
            uk_data = json.load(f)
            print(f"Loaded {len(uk_data)} UK articles")
            # Add UK to the mix if we want 'all' to be truly all
            # But for now let's just keep the scraped ones in the main list or combine everything?
            # Let's combine everything in all_articles.json
            
    all_articles = []
    for data in codes.values():
        all_articles.extend(data)
        
    if os.path.exists(uk_path):
         all_articles.extend(uk_data)
         
    with open(os.path.join(output_dir, "all_articles.json"), "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
    print(f"Saved total {len(all_articles)} articles to all_articles.json")

if __name__ == "__main__":
    merge_data()
