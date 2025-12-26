# Архитектура проекта Legal NLP

Документация архитектуры системы для работы с российским законодательством.

## Общая архитектура системы

```mermaid
graph TB
    subgraph "Источники данных"
        A1[zakonrf.info]
        A2[sudact.ru]
        A3[actual.pravo.gov.ru]
    end
    
    subgraph "Сбор данных"
        B1[Web Scrapers]
        B2[BaseStrategy Pattern]
        B3[Dynamic Scraping<br/>Playwright]
    end
    
    subgraph "Хранение сырых данных"
        C1[data/raw/<br/>JSON files]
    end
    
    subgraph "Обработка данных"
        D1[Ingest<br/>Загрузка]
        D2[Normalize<br/>Нормализация]
        D3[Clean<br/>Очистка]
        D4[Quality Filters<br/>Фильтры качества]
        D5[Deduplication<br/>Дедупликация]
        D6[Split<br/>Разделение]
    end
    
    subgraph "Хранение обработанных данных"
        E1[train.jsonl]
        E2[val.jsonl]
        E3[test.jsonl]
    end
    
    subgraph "RAG Pipeline"
        F1[ChromaDB<br/>Vector Store]
        F2[Sentence Transformers<br/>Embeddings]
        F3[Semantic Search]
    end
    
    subgraph "Fine-tuning Pipeline"
        G1{Backend?}
        G2[MLX<br/>Apple Silicon]
        G3[CUDA<br/>NVIDIA GPU]
        G4[QLoRA<br/>4-bit]
        G5[LoRA Adapters]
    end
    
    subgraph "Обученные модели"
        H1[Legal Assistant<br/>MLX Model]
        H2[Legal Assistant<br/>CUDA Model]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    
    C1 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5
    D5 --> D6
    D6 --> E1
    D6 --> E2
    D6 --> E3
    
    E1 --> G1
    G1 -->|Apple Silicon| G2
    G1 -->|NVIDIA GPU| G3
    G2 --> G4
    G3 --> G4
    G4 --> G5
    G5 --> H1
    G5 --> H2
    
    C1 --> F1
    F1 --> F2
    F2 --> F3
```

## Pipeline предобработки данных

```mermaid
flowchart TD
    Start([Начало]) --> Load[Загрузка сырых данных<br/>data/raw/*.json]
    
    Load --> Format{Определение<br/>формата}
    
    Format -->|case/article| Parse1[Извлечение case и article]
    Format -->|messages| Parse2[Использование messages]
    Format -->|title/content| Parse3[Создание Q&A пар]
    
    Parse1 --> Normalize[Нормализация к ChatML<br/>messages структуре]
    Parse2 --> Normalize
    Parse3 --> Normalize
    
    Normalize --> Clean[Очистка текста<br/>- HTML теги<br/>- Артефакты<br/>- Нормализация пробелов]
    
    Clean --> Filter[Фильтры качества<br/>- Минимальная длина<br/>- Проверка на мусор<br/>- Обнаружение обрывов]
    
    Filter --> Dedup[Дедупликация<br/>- Exact duplicates<br/>- Near-duplicates]
    
    Dedup --> Group[Группировка по статьям<br/>group_id = code_article_num]
    
    Group --> Stratify[Стратификация<br/>по кодексам]
    
    Stratify --> Split[Group Split<br/>80% train / 10% val / 10% test<br/>без утечек]
    
    Split --> Save1[Сохранение train.jsonl]
    Split --> Save2[Сохранение val.jsonl]
    Split --> Save3[Сохранение test.jsonl]
    Split --> Manifest[Сохранение split_manifest.json]
    Split --> Report[Генерация preprocessing_report.json]
    
    Save1 --> End([Конец])
    Save2 --> End
    Save3 --> End
    Manifest --> End
    Report --> End
```

## RAG System Architecture

```mermaid
graph LR
    subgraph "Input"
        A[Legal Articles<br/>JSON format]
    end
    
    subgraph "Processing"
        B[Chunking<br/>2500 chars<br/>400 overlap]
        C[Sentence Transformer<br/>multilingual-e5-large-instruct]
        D[Generate Embeddings<br/>768 dimensions]
    end
    
    subgraph "Storage"
        E[ChromaDB<br/>PersistentClient]
        F[Collection: russian_law<br/>- documents<br/>- embeddings<br/>- metadata]
    end
    
    subgraph "Query"
        G[User Query<br/>Text]
        H[Query Embedding]
        I[Similarity Search<br/>Cosine Distance]
        J[Top-K Results<br/>with metadata]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    G --> H
    H --> I
    F --> I
    I --> J
```

## Fine-tuning Pipeline

```mermaid
flowchart TD
    Start([Начало обучения]) --> Detect{Определение<br/>бэкенда}
    
    Detect -->|Apple Silicon| MLX[MLX Backend]
    Detect -->|NVIDIA GPU| CUDA[CUDA Backend]
    Detect -->|CPU| Error[Ошибка:<br/>требуется GPU]
    
    subgraph "MLX Pipeline"
        MLX --> MLXLoad[mlx_lm.lora.load<br/>Загрузка модели]
        MLXLoad --> MLXData[ChatDataset<br/>Загрузка данных]
        MLXData --> MLXConfig[LoRA Config<br/>rank=16, alpha=32]
        MLXConfig --> MLXTrain[MLX Training<br/>2000 iterations]
        MLXTrain --> MLXSave[Сохранение<br/>adapters.safetensors]
    end
    
    subgraph "CUDA Pipeline"
        CUDA --> CUDALoad[AutoModelForCausalLM<br/>from_pretrained<br/>4-bit quantization]
        CUDALoad --> CUDAPEFT[PEFT LoRA<br/>rank=64, alpha=32]
        CUDAPEFT --> CUDAData[Dataset<br/>Загрузка данных]
        CUDAData --> CUDATokenize[Tokenization<br/>ChatML format]
        CUDATokenize --> CUDATrain[Trainer<br/>3 epochs]
        CUDATrain --> CUDASave[Сохранение<br/>LoRA adapters]
    end
    
    MLXSave --> End([Конец])
    CUDASave --> End
    Error --> End
```

## Компоненты системы

### 1. Web Scrapers

```mermaid
classDiagram
    class BaseStrategy {
        <<abstract>>
        +get_article_links(start_url, limit) List[str]
        +parse_article(url, html) Dict
    }
    
    class ZakonRFStrategy {
        +get_article_links(start_url, limit)
        +parse_article(url, html)
    }
    
    class SudactStrategy {
        +get_article_links(start_url, limit)
        +parse_article(url, html)
    }
    
    class SmartScraper {
        +parse_codex_hashes()
        +extract_redid()
        +api_getcontent()
        +api_redtext()
        +build_article_record()
    }
    
    BaseStrategy <|-- ZakonRFStrategy
    BaseStrategy <|-- SudactStrategy
    SmartScraper ..> BaseStrategy : uses pattern
```

### 2. Data Preprocessing

```mermaid
classDiagram
    class DataPreprocessor {
        -seed: int
        -stats: ProcessingStats
        +ingest_raw(raw_dir) List[Dict]
        +normalize_format(records) List[Dict]
        +apply_text_cleaning(records) List[Dict]
        +quality_filters(records) List[Dict]
        +deduplication(records) List[Dict]
        +split_train_val_test(records) Tuple
        +export_jsonl(records, path)
        +generate_report(output_dir)
    }
    
    class ProcessingStats {
        +initial_count: int
        +after_normalization: int
        +after_cleaning: int
        +after_quality_filters: int
        +after_deduplication: int
        +train_count: int
        +val_count: int
        +test_count: int
        +removed_reasons: Dict
    }
    
    DataPreprocessor --> ProcessingStats
```

### 3. RAG System

```mermaid
classDiagram
    class RAGSystem {
        +build_vector_db(articles_path, db_path)
        +search_legal_context(query, db_path, top_k)
    }
    
    class ChromaDB {
        +PersistentClient(path)
        +create_collection(name)
        +query(query_texts, n_results)
    }
    
    class SentenceTransformer {
        +encode(text, normalize_embeddings)
    }
    
    RAGSystem --> ChromaDB
    RAGSystem --> SentenceTransformer
```

### 4. Fine-tuning System

```mermaid
classDiagram
    class FineTuner {
        <<abstract>>
        +load_model(model_id)
        +load_data(data_path)
        +train()
        +save(output_dir)
    }
    
    class MLXFineTuner {
        +load_model(model_id)
        +train()
        +save()
    }
    
    class CUDAFineTuner {
        +load_model(model_id)
        +train()
        +save()
    }
    
    FineTuner <|-- MLXFineTuner
    FineTuner <|-- CUDAFineTuner
```

## Потоки данных

### Поток 1: Сбор и обработка данных

```mermaid
sequenceDiagram
    participant User
    participant Scraper
    participant Processor
    participant Storage
    
    User->>Scraper: Запуск скрапера
    Scraper->>Scraper: Сбор данных с сайтов
    Scraper->>Storage: Сохранение в data/raw/
    
    User->>Processor: Запуск предобработки
    Processor->>Storage: Чтение из data/raw/
    Processor->>Processor: Нормализация формата
    Processor->>Processor: Очистка текста
    Processor->>Processor: Фильтры качества
    Processor->>Processor: Дедупликация
    Processor->>Processor: Разделение на train/val/test
    Processor->>Storage: Сохранение в data/processed/
```

### Поток 2: Построение RAG системы

```mermaid
sequenceDiagram
    participant User
    participant RAG
    participant Embedder
    participant ChromaDB
    
    User->>RAG: build_vector_db()
    RAG->>RAG: Загрузка статей из JSON
    RAG->>RAG: Чанкинг длинных статей
    
    loop Для каждого чанка
        RAG->>Embedder: encode(chunk)
        Embedder->>RAG: embedding vector
        RAG->>ChromaDB: add(document, embedding, metadata)
    end
    
    RAG->>User: База данных создана
```

### Поток 3: Поиск в RAG системе

```mermaid
sequenceDiagram
    participant User
    participant RAG
    participant Embedder
    participant ChromaDB
    
    User->>RAG: search_legal_context(query)
    RAG->>Embedder: encode(query)
    Embedder->>RAG: query_embedding
    
    RAG->>ChromaDB: query(query_embedding, top_k=5)
    ChromaDB->>RAG: results with metadata
    
    RAG->>RAG: Форматирование результатов
    RAG->>User: Список релевантных статей
```

### Поток 4: Fine-tuning

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant Model
    participant Data
    
    User->>Trainer: Запуск обучения
    Trainer->>Model: Загрузка базовой модели
    Trainer->>Data: Загрузка train.jsonl
    
    loop Эпохи/Итерации
        Trainer->>Data: Получение батча
        Trainer->>Model: Forward pass
        Model->>Trainer: Loss
        Trainer->>Model: Backward pass
        Trainer->>Model: Обновление весов LoRA
    end
    
    Trainer->>Trainer: Сохранение адаптеров
    Trainer->>User: Обучение завершено
```

## Структура данных

### Формат входных данных (сырые)

```mermaid
graph TD
    A[Сырые данные] --> B[Формат 1:<br/>case/article]
    A --> C[Формат 2:<br/>messages]
    A --> D[Формат 3:<br/>title/content]
    
    B --> E{JSON объект}
    C --> E
    D --> E
```

### Формат выходных данных (для обучения)

```mermaid
graph TD
    A[Обработанные данные] --> B[ChatML формат]
    B --> C[messages: Array]
    C --> D[system: System prompt]
    C --> E[user: Вопрос]
    C --> F[assistant: Ответ]
    
    D --> G[JSONL файл]
    E --> G
    F --> G
```

## Метаданные и конфигурация

### Метаданные статей в RAG

```mermaid
graph LR
    A[Article Metadata] --> B[article_url]
    A --> C[article_title]
    A --> D[legal_code]
    A --> E[chunk_start]
    A --> F[chunk_end]
```

### Конфигурация LoRA

```mermaid
graph TD
    A[LoRA Config] --> B[rank: 16/64]
    A --> C[alpha: 32]
    A --> D[dropout: 0.05]
    A --> E[target_modules]
    
    E --> F[q_proj]
    E --> G[k_proj]
    E --> H[v_proj]
    E --> I[o_proj]
    E --> J[gate_proj]
    E --> K[up_proj]
    E --> L[down_proj]
```

## Масштабирование и производительность

### Стратегии оптимизации

```mermaid
graph TD
    A[Оптимизация] --> B[Память]
    A --> C[Скорость]
    A --> D[Качество]
    
    B --> B1[4-bit Quantization]
    B --> B2[Gradient Checkpointing]
    B --> B3[Batch Size Tuning]
    
    C --> C1[Unsloth для CUDA]
    C --> C2[MLX для Apple Silicon]
    C --> C3[Efficient Embeddings]
    
    D --> D1[Group Split]
    D --> D2[Quality Filters]
    D --> D3[Stratification]
```

---

**Версия документа**: 1.0  
**Последнее обновление**: 2024

