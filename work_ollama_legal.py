import json
import requests
from pathlib import Path
import time

# Настройки
OLLAMA_URL = "http://localhost:11434/api/generate"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

class OllamaLegal:
    def __init__(self, model="mistral:7b-instruct"):
        self.model = model
        self.url = OLLAMA_URL
        
    def query(self, prompt, max_tokens=300, timeout=30):
        """Запрос к Ollama с расширенным таймаутом"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.url, 
                json=data, 
                timeout=(10, timeout),  # (connect, read)
                headers={'Connection': 'keep-alive'}
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Ошибка HTTP {response.status_code}: {response.text[:100]}"
                
        except requests.exceptions.Timeout:
            return "Ошибка: Запрос превысил время ожидания"
        except requests.exceptions.ConnectionError:
            return "Ошибка: Не удалось подключиться к Ollama"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def legal_consult(self, question, context="", max_tokens=300):
        """Юридическая консультация"""
        system_prompt = "Вы - юридический консультант РФ. Дайте точный ответ на основе законодательства."
        
        if context:
            prompt = f"{system_prompt}\n\nКонтекст: {context[:500]}...\n\nВопрос: {question}\n\nОтвет:"
        else:
            prompt = f"{system_prompt}\n\nВопрос: {question}\n\nОтвет:"
            
        return self.query(prompt, max_tokens=max_tokens)

def test_basic_questions():
    """Тест базовых вопросов"""
    legal = OllamaLegal("mistral:7b-instruct")
    
    questions = [
        "Что такое трудовой договор по ТК РФ?",
        "Какие виды ответственности в ГК РФ?",
        "Что такое исковая давность?"
    ]
    
    print("🤖 Тест базовых юридических вопросов")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q}")
        print("-" * 40)
        
        answer = legal.legal_consult(q, max_tokens=200)
        print(f"Ответ: {answer[:300]}...")
        
        time.sleep(2)  # Задержка между запросами

def test_with_dataset():
    """Тест с датасетом"""
    try:
        with open(DATA_DIR / "processed" / "synthetic_qa_labeled.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Берем первые 3 записи
        samples = data[:3]
        legal = OllamaLegal("deepseek-r1:latest")
        
        print("\n📚 Тест с использованием датасета")
        print("=" * 60)
        
        for i, sample in enumerate(samples, 1):
            question = sample.get("question", "Что такое трудовой договор?")
            context = sample.get("context", "")
            
            print(f"\n{i}. Вопрос из датасета: {question}")
            print(f"   Контекст: {context[:100]}...")
            print("-" * 40)
            
            # Короткий ответ
            answer = legal.legal_consult(question, context, max_tokens=150)
            print(f"Ответ: {answer}")
            
            time.sleep(3)  # Задержка между запросами
            
    except Exception as e:
        print(f"Ошибка при работе с датасетом: {e}")

def compare_models():
    """Сравнение моделей на одном вопросе"""
    models = ["mistral:7b-instruct", "llama3.2:latest"]
    test_question = "Что такое трудовой договор по ТК РФ?"
    
    print("\n🔄 Сравнение моделей")
    print("=" * 60)
    print(f"Вопрос: {test_question}")
    print("=" * 60)
    
    for model in models:
        print(f"\n🤖 Модель: {model}")
        print("-" * 30)
        
        legal = OllamaLegal(model)
        answer = legal.legal_consult(test_question, max_tokens=200)
        print(f"Ответ: {answer}")
        
        time.sleep(2)

def main():
    print("🚀 Запуск рабочего теста Ollama...")
    print("⚠️  Внимание: запросы могут занимать время, ждем ответов...")
    
    # Тест 1: Базовые вопросы
    test_basic_questions()
    
    # Тест 2: С датасетом
    test_with_dataset()
    
    # Тест 3: Сравнение моделей
    compare_models()
    
    print("\n✅ Все тесты завершены!")

if __name__ == "__main__":
    main()