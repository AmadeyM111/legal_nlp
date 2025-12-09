import json
import requests
from pathlib import Path

# Диагностика файла данных
def check_data_file():
    data_path = Path("../data/processed/synthetic_qa_labeled.json")
    print(f"Проверка файла: {data_path}")
    
    if data_path.exists():
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Файл открыт, размер: {len(content)} символов")
                
                # Проверяем начало файла
                if content.startswith('[') or content.startswith('{'):
                    print("✅ Формат JSON корректный")
                    
                    # Пробуем распарсить
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            print(f"✅ JSON распарсен, записей: {len(data)}")
                            if data:
                                first_item = data[0]
                                print(f"📄 Первая запись: {first_item}")
                        return True
                    except json.JSONDecodeError as e:
                        print(f"❌ Ошибка JSON: {e}")
                        return False
                else:
                    print(f"❌ Файл начинается не с JSON: {content[:50]}...")
                    return False
                    
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {e}")
            return False
    else:
        print("❌ Файл не найден")
        return False

# Диагностика Ollama
def check_ollama():
    print("\nПроверка Ollama...")
    try:
        # Простой GET запрос
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama API доступен")
            models = response.json().get('models', [])
            print(f"📦 Моделей найдено: {len(models)}")
            return True
        else:
            print(f"❌ Ollama вернул статус: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения к Ollama: {e}")
        return False

# Диагностика POST запроса
def test_ollama_post():
    print("\nТест POST запроса к Ollama...")
    try:
        data = {
            "model": "mistral:7b-instruct",
            "prompt": "Тест",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=data, 
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print("✅ POST запрос работает")
                return True
            else:
                print(f"❌ Неверный формат ответа: {result}")
                return False
        else:
            print(f"❌ POST запрос вернул статус: {response.status_code}")
            print(f"Текст ошибки: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка POST запроса: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Диагностика возможных причин EOF ошибки")
    print("=" * 50)
    
    # Проверяем файл данных
    data_ok = check_data_file()
    
    # Проверяем Ollama
    ollama_ok = check_ollama()
    
    if ollama_ok:
        # Тестируем POST запрос
        post_ok = test_ollama_post()
        
        if post_ok:
            print("\n✅ Все проверки пройдены!")
            print("Проблема EOF, вероятно, в использовании input() или в разрыве соединения при длительных запросах")
        else:
            print("\n❌ Проблема с POST запросами к Ollama")
    
    print("\n🔧 Рекомендации:")
    print("1. Избегайте input() в скриптах для background выполнения")
    print("2. Используйте timeout в HTTP запросах")
    print("3. Проверяйте完整性 JSON файлов перед загрузкой")