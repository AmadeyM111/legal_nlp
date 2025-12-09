# Legal NLP Project Summary

## Project Overview
This is a Legal RAG (Retrieval-Augmented Generation) system designed for Russian legal documents. The project uses the Saiga-Mistral 7B model, which is a Russian-language fine-tuned model, to provide legal assistance and answer questions related to Russian law.

## Key Components

### 1. Model
- **Base Model**: IlyaGusev/saiga_mistral_7b_merged
- **Purpose**: Russian legal language understanding and generation
- **Model Size**: 7B parameters
- **Location**: models/saiga_mistral_7b_merged/

### 2. Data Processing
- **Raw Data**: Legal articles scraped from Russian legal websites (zakonrf.info)
- **Processing Pipeline**: 
  - Scraping legal documents from various Russian legal codes
  - Cleaning and preprocessing of legal text
  - Statistical analysis of the legal corpus
- **Legal Codes Covered**: Includes major Russian legal codes (GK, NK, TK, JK, UK - Гражданский, Налоговый, Трудовой, Жилищный, Уголовный)

### 3. Core Functionality
- **RAG System**: Uses ChromaDB for vector storage and retrieval of legal documents
- **Legal QA**: Fine-tuned models for answering legal questions
- **Evaluation**: Built-in evaluation framework for assessing legal response quality

### 4. Model Training
- **Fine-tuning**: LoRA and QLoRA fine-tuning approaches for legal domain adaptation
- **Output Models**: 
  - models/legal-saiga-lora
  - models/saiga-legal-7b
  - models/saiga_legal_final

## File Hash Verification
The project includes a hash verification system (`scripts/compute_hashes.py`) to ensure model file integrity, which is crucial for legal applications where model consistency is important.

## Applications
- Legal document search and retrieval
- Legal question answering
- Legal text generation and summarization
- Compliance checking against Russian legal codes

## Technical Stack
- Python 3.11+
- Transformers library
- PyTorch
- PEFT (Parameter Efficient Fine-Tuning)
- ChromaDB for vector storage
- Hugging Face ecosystem

## Usage
The system can be used to:
1. Query legal documents using natural language
2. Retrieve relevant legal articles based on user questions
3. Generate legal responses based on the Russian legal corpus
4. Evaluate legal responses for accuracy and relevance

## Model Integrity
The project emphasizes model integrity through hash verification to ensure that the legal models have not been tampered with, which is crucial for legal applications requiring trust and reliability.