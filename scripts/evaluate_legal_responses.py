import json
import requests
from pathlib import Path
import re
import time

# Настройки
OLLAMA_URL = "http://localhost:11434/api/generate"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

class LegalEvaluator:
    def __init__(self, model="mistral:7b-instruct"):
        self.model = model
        self.url = OLLAMA_URL
        
    def query_model(self, question, temperature=0.3):
        """Запрос к модели"""
        data = {
            "model": self.model,
            "prompt": f"Вы - юрист-консультант РФ. Ответьте на вопрос: {question}",
            "stream": False,
            "options": {
                "temperature": temperature,
                "max_tokens": 500
            }
        }
        
        response = requests.post(self.url, json=data)
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return None
    
    def evaluate_answer(self, question, model_answer, reference_answer=""):
        """Оценка ответа по критериям"""
        scores = {
            "relevance": 0,
            "completeness": 0,
            "accuracy": 0,
            "structure": 0
        }
        
        # Релевантность (содержит ли ответ ключевые термины из вопроса)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', model_answer.lower()))
        relevance = len(question_words & answer_words) / len(question_words) if question_words else 0
        scores["relevance"] = min(relevance * 100, 100)
        
        # Полнота (количество предложений)
        sentences = model_answer.split('.')
        completeness = min(len([s for s in sentences if len(s.strip()) > 20]) * 20, 100)
        scores["completeness"] = completeness
        
        # Точность (проверка на юридические термины)
        legal_terms = ["кодекс", "статья", "закон", "пункт", "положение", "ответственность", "право", "обязанность"]
        accuracy = sum(1 for term in legal_terms if term in model_answer.lower()) * 10
        scores["accuracy"] = min(accuracy, 100)
        
        # Структура (наличие вступления, основной части, заключения)
        has_intro = len(model_answer) > 50
        has_main = len(model_answer) > 150
        has_conclusion = len(model_answer) > 250
        structure = (has_intro + has_main + has_conclusion) * 33.33
        scores["structure"] = min(structure, 100)
        
        return scores
    
    def test_legal_questions(self):
        """Тестирование на юридических вопросах"""
        test_questions = [
            {
                "question": "Что такое трудовой договор по ТК РФ и какие его существенные условия?",
                "category": "ТК РФ",
                "expected_keywords": ["трудовой договор", "условия", "ТК", "работник", "работодатель"]
            },
            {
                "question": "Какие виды ответственности предусмотрены в Гражданском кодексе РФ?",
                "category": "ГК РФ", 
                "expected_keywords": ["ответственность", "ГК", "обязательства", "вред", "убытки"]
            },
            {
                "question": "Что такое исковая давность и какой ее срок по общему правилу?",
                "category": "ГК РФ",
                "expected_keywords": ["исковая давность", "срок", "три года", "общее правило"]
            }
        ]
        
        print("🔍 Оценка качества юридических ответов")
        print("=" * 70)
        print(f"Модель: {self.model}")
        print("=" * 70)
        
        results = []
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n📝 Тест {i}: {test['category']}")
            print(f"❓ Вопрос: {test['question']}")
            print("-" * 50)
            
            # Запрос к модели
            answer = self.query_model(test['question'])
            
            if answer:
                print(f"🤖 Ответ модели:\n{answer}\n")
                
                # Оценка ответа
                scores = self.evaluate_answer(test['question'], answer)
                avg_score = sum(scores.values()) / len(scores)
                
                print("📊 Оценка:")
                for criterion, score in scores.items():
                    print(f"   • {criterion}: {score:.1f}%")
                print(f"   📈 Средний балл: {avg_score:.1f}%")
                
                # Проверка на ожидаемые ключевые слова
                found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in answer.lower()]
                keyword_coverage = len(found_keywords) / len(test['expected_keywords']) * 100
                
                print(f"   🔑 Найдено ключевых слов: {len(found_keywords)}/{len(test['expected_keywords'])} ({keyword_coverage:.1f}%)")
                print(f"   🔍 Слова: {', '.join(found_keywords)}")
                
                results.append({
                    "test": i,
                    "category": test['category'],
                    "question": test['question'],
                    "answer": answer,
                    "scores": scores,
                    "avg_score": avg_score,
                    "keyword_coverage": keyword_coverage
                })
            else:
                print("❌ Не удалось получить ответ от модели")
                results.append({
                    "test": i,
                    "category": test['category'],
                    "question": test['question'],
                    "answer": None,
                    "error": True
                })
            
            time.sleep(1)  # Задержка между запросами
        
        # Итоговая статистика
        print("\n" + "=" * 70)
        print("📈 ОБЩАЯ СТАТИСТИКА")
        print("=" * 70)
        
        valid_results = [r for r in results if not r.get('error')]
        
        if valid_results:
            avg_scores = {
                criterion: sum(r['scores'][criterion] for r in valid_results) / len(valid_results)
                for criterion in valid_results[0]['scores']
            }
            avg_overall = sum(r['avg_score'] for r in valid_results) / len(valid_results)
            avg_keywords = sum(r['keyword_coverage'] for r in valid_results) / len(valid_results)
            
            print(f"📊 Средние оценки по критериям:")
            for criterion, score in avg_scores.items():
                print(f"   • {criterion}: {score:.1f}%")
            print(f"   🎯 Общий средний балл: {avg_overall:.1f}%")
            print(f"   🔑 Среднее покрытие ключевых слов: {avg_keywords:.1f}%")
        
        # Сохранение результатов
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = DATA_DIR / f"evaluation_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": self.model,
                "timestamp": timestamp,
                "results": results,
                "summary": {
                    "total_tests": len(test_questions),
                    "successful_tests": len(valid_results),
                    "avg_overall_score": avg_overall if valid_results else 0,
                    "avg_keyword_coverage": avg_keywords if valid_results else 0
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в: {output_file}")
        return results

if __name__ == "__main__":
    print("🚀 Запуск оценки юридических моделей...")
    
    # Создаем экземпляр оценщика
    evaluator = LegalEvaluator("mistral:7b-instruct")
    
    # Запускаем тестирование
    results = evaluator.test_legal_questions()
    
    print("\n✅ Оценка завершена!")