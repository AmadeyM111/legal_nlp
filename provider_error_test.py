import requests
import json
import time
import subprocess
from pathlib import Path

def test_model_health():
    """Проверяем здоровье моделей"""
    print("🔍 Тест здоровья моделей Ollama")
    print("=" * 50)
    
    # Проверяем список моделей
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Найдено моделей: {len(models)}")
            
            for model in models:
                name = model['name']
                size = model['size']
                size_gb = size / (1024**3)
                print(f"  📦 {name}: {size_gb:.1f}GB")
        else:
            print(f"❌ Ошибка получения моделей: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False
    
    return True

def test_individual_models():
    """Тестируем каждую модель по отдельности"""
    test_prompt = "Ответь одним словом: тест"
    
    models_to_test = [
        "llama3.2:latest",  # Самая маленькая (2GB)
        "mistral:7b-instruct",  # Средняя (4.4GB)
        "deepseek-r1:7b",  # Средняя (4.7GB)
        "deepseek-r1:latest"  # Самая большая (5.2GB)
    ]
    
    print("\n🧪 Тестирование отдельных моделей")
    print("=" * 50)
    
    results = {}
    
    for model in models_to_test:
        print(f"\n🤖 Тестируем модель: {model}")
        print("-" * 30)
        
        try:
            # Останавливаем все модели
            subprocess.run(['ollama', 'stop'], capture_output=True)
            time.sleep(1)
            
            # Запускаем тест
            data = {
                "model": model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "num_ctx": 512,  # Маленький контекст
                    "num_batch": 1,
                    "temperature": 0
                }
            }
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate", 
                json=data, 
                timeout=15
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                results[model] = {
                    "success": True,
                    "time": elapsed,
                    "answer": answer
                }
                print(f"✅ Успех ({elapsed:.1f}s): {answer}")
            else:
                error_text = response.text[:200]
                results[model] = {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": error_text
                }
                print(f"❌ Ошибка HTTP {response.status_code}")
                print(f"   {error_text}")
                
        except requests.exceptions.Timeout:
            results[model] = {
                "success": False,
                "error": "Timeout"
            }
            print("❌ Таймаут запроса")
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Ошибка: {e}")
        
        time.sleep(2)  # Пауза между тестами
    
    return results

def analyze_results(results):
    """Анализируем результаты тестов"""
    print("\n📊 Анализ результатов")
    print("=" * 50)
    
    successful = [m for m, r in results.items() if r.get('success')]
    failed = [m for m, r in results.items() if not r.get('success')]
    
    print(f"✅ Работающие модели: {len(successful)}")
    for model in successful:
        r = results[model]
        print(f"  📦 {model}: {r['time']:.1f}s")
    
    if failed:
        print(f"\n❌ Не работающие модели: {len(failed)}")
        for model in failed:
            r = results[model]
            print(f"  📦 {model}: {r['error']}")
            if 'details' in r:
                print(f"     Детали: {r['details'][:100]}...")
    
    # Рекомендации
    print("\n💡 Рекомендации:")
    
    if len(successful) == 0:
        print("⚠️  Ни одна модель не работает. Проверьте:")
        print("   1. Достаточно ли RAM (нужно минимум 8GB свободной)")
        print("   2. Не блокирует ли антивирус Ollama")
        print("   3. Перезапустите Ollama: ollama serve")
    elif len(successful) < len(results):
        print("⚠️  Некоторые модели не работают. Используйте рабочие.")
        print("   Обычно меньшие модели (llama3.2) работают стабильнее")
    else:
        print("✅ Все модели работают отлично!")
    
    return successful[0] if successful else None

def test_problematic_model(model_name):
    """Детальное тестирование проблемной модели"""
    print(f"\n🔬 Детальное тестирование модели: {model_name}")
    print("=" * 50)
    
    tests = [
        {"prompt": "тест", "tokens": 10, "name": "Короткий запрос"},
        {"prompt": "Что такое закон?", "tokens": 50, "name": "Средний запрос"},
        {"prompt": "Расскажите подробно о гражданском праве", "tokens": 200, "name": "Длинный запрос"}
    ]
    
    for test in tests:
        print(f"\n📝 {test['name']}: {test['prompt']}")
        print("-" * 30)
        
        try:
            data = {
                "model": model_name,
                "prompt": test['prompt'],
                "stream": False,
                "options": {
                    "max_tokens": test['tokens'],
                    "temperature": 0
                }
            }
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate", 
                json=data, 
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                print(f"✅ Успех ({elapsed:.1f}s): {answer[:100]}...")
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"   {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Исключение: {e}")

def main():
    print("🔧 Диагностика Provider Error в Ollama")
    print("=" * 50)
    
    # Шаг 1: Базовая проверка
    if not test_model_health():
        return
    
    # Шаг 2: Тестируем модели
    results = test_individual_models()
    
    # Шаг 3: Анализ
    working_model = analyze_results(results)
    
    # Шаг 4: Детальное тестирование проблемных моделей
    for model, result in results.items():
        if not result.get('success'):
            test_problematic_model(model)
            break
    
    print("\n" + "=" * 50)
    print("🏁 Диагностика завершена!")

if __name__ == "__main__":
    main()