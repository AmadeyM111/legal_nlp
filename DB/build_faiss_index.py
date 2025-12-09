import json
import argparse
import os
import numpy as np
import faiss

def build_faiss_index(articles, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """
    Builds FAISS index from article contents.
    
    Returns:
        - index: FAISS index
        - metadata: List of dicts with article info for each chunk
    """
    # Import here to avoid lzma dependency issues
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Extracting chunks and creating embeddings...")
    chunks = []
    metadata = []
    
    for article in articles:
        content = article.get('content', '').strip()
        if not content:
            continue
        
        # Use the existing content as-is (already cleaned)
        # We can optionally chunk it further, but for now use full content if < 2500 chars
        # For larger articles, we could split them
        
        if len(content) > 2500:
            # Split into chunks of 2500 with 400 overlap
            chunk_size = 2500
            overlap = 400
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk = content[start:end]
                chunks.append(chunk)
                metadata.append({
                    'article_url': article['url'],
                    'article_title': article['title'],
                    'chunk_start': start,
                    'chunk_end': end
                })
                if end == len(content):
                    break
                start += chunk_size - overlap
        else:
            chunks.append(content)
            metadata.append({
                'article_url': article['url'],
                'article_title': article['title'],
                'chunk_start': 0,
                'chunk_end': len(content)
            })
    
    print(f"Total chunks: {len(chunks)}")
    
    # Create embeddings
    print("Creating embeddings (this may take a few minutes)...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings.astype('float32'))
    
    print(f"Index built with {index.ntotal} vectors")
    
    return index, metadata, chunks

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG system.")
    parser.add_argument("--input", type=str, default="data/processed/articles_cleaned.json", help="Input cleaned articles file")
    parser.add_argument("--output_dir", type=str, default="data/faiss_index/", help="Output directory")
    args = parser.parse_args()

    print(f"Loading articles from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles")
    
    # Build index
    index, metadata, chunks = build_faiss_index(articles)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    index_path = os.path.join(args.output_dir, "index.faiss")
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    chunks_path = os.path.join(args.output_dir, "chunks.json")
    
    faiss.write_index(index, index_path)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Saved FAISS index to {index_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved chunks to {chunks_path}")

if __name__ == "__main__":
    main()
