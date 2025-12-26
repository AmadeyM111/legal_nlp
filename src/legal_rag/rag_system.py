import os
import json
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def build_vector_db(articles_path="data/processed/articles_cleaned.json", db_path="./chroma_legal_db"):
    """
    Builds ChromaDB vector store from cleaned articles.
    """
    logger.info("Загрузка эмбеддера (multilingual-e5-large-instruct)...")
    embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    
    logger.info(f"Загрузка статей из {articles_path}...")
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    logger.info(f"Создание ChromaDB в {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Удаляем старую коллекцию если есть
    try:
        client.delete_collection(name="russian_law")
    except Exception as e:
        logger.debug(f"Коллекция не существовала или ошибка удаления: {e}")
    
    collection = client.create_collection(
        name="russian_law",
        metadata={"description": "Russian legal articles with embeddings"}
    )
    
    logger.info("Подготовка чанков и эмбеддингов...")
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    chunk_id = 0
    for article in tqdm(articles, desc="Обработка статей"):
        content = article.get('content', '').strip()
        if not content or len(content) < 200:
            continue
        
        # Чанкуем длинные статьи (как в generate_synthetic_data.py)
        if len(content) > 2500:
            chunk_size = 2500
            overlap = 400
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk = content[start:end]
                
                if len(chunk) >= 200:  # Не сохраняем слишком короткие
                    embedding = embedder.encode(chunk, normalize_embeddings=True)
                    
                    documents.append(chunk)
                    metadatas.append({
                        "article_url": article['url'],
                        "article_title": article['title'],
                        "chunk_start": start,
                        "chunk_end": end,
                        "legal_code": article['url'].split('/')[3].upper()  # GK, NK, TK, JK, UK
                    })
                    ids.append(f"chunk_{chunk_id:06d}")
                    embeddings.append(embedding.tolist())
                    chunk_id += 1
                
                if end == len(content):
                    break
                start += chunk_size - overlap
        else:
            # Статья целиком
            embedding = embedder.encode(content, normalize_embeddings=True)
            
            documents.append(content)
            metadatas.append({
                "article_url": article['url'],
                "article_title": article['title'],
                "chunk_start": 0,
                "chunk_end": len(content),
                "legal_code": article['url'].split('/')[3].upper()
            })
            ids.append(f"chunk_{chunk_id:06d}")
            embeddings.append(embedding.tolist())
            chunk_id += 1
    
    logger.info(f"Добавление {len(documents)} чанков в ChromaDB...")
    # Добавляем батчами по 100 (ChromaDB может глючить на больших батчах)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        logger.info(f"  Добавлено {batch_end}/{len(documents)} чанков")
    
    logger.info(f"✅ Готово! Сохранено {len(documents)} чанков в {db_path}")
    return client, collection

def search_legal_context(query: str, db_path="./chroma_legal_db", top_k=5):
    """
    Searches for relevant legal articles using ChromaDB.
    """
    embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="russian_law")
    
    # Поиск
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Форматируем результаты
    formatted_results = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        formatted_results.append({
            "text": doc,
            "article_title": metadata['article_title'],
            "article_url": metadata['article_url'],
            "legal_code": metadata['legal_code']
        })
    
    return formatted_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build or query ChromaDB RAG system")
    parser.add_argument("--build", action="store_true", help="Build the vector database")
    parser.add_argument("--query", type=str, help="Query the database")
    parser.add_argument("--articles", type=str, default="data/processed/articles_cleaned.json")
    parser.add_argument("--db_path", type=str, default="./chroma_legal_db")
    args = parser.parse_args()
    
    if args.build:
        build_vector_db(args.articles, args.db_path)
    elif args.query:
        logger.info(f"\nВопрос: {args.query}\n")
        results = search_legal_context(args.query, args.db_path)
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['article_title']}")
            logger.info(f"   Код: {result['legal_code']}")
            logger.info(f"   Текст: {result['text'][:200]}...")
            logger.info("")
    else:
        logger.info("Usage: python src/rag_system.py --build  OR  python src/rag_system.py --query 'ваш вопрос'")
