import json
import os
import random

MOCK_CASES = [
    {
        "situation": "Работодатель не выплатил заработную плату за два месяца, ссылаясь на отсутствие денег на счетах компании.",
        "article": "Статья 142 ТК РФ",
        "code": "ТК РФ",
        "reasoning": "Задержка выплаты заработной платы более 15 дней позволяет работнику приостановить работу."
    },
    {
        "situation": "Покупатель обнаружил существенный недостаток в купленном автомобиле через 10 дней после покупки и требует возврата денег.",
        "article": "Статья 18 ЗоЗПП",
        "code": "ЗоЗПП",
        "reasoning": "Потребитель имеет право на возврат товара при обнаружении недостатков."
    },
    {
        "situation": "Арендатор испортил имущество в съемной квартире и отказывается возмещать ущерб.",
        "article": "Статья 1064 ГК РФ",
        "code": "ГК РФ",
        "reasoning": "Вред, причиненный личности или имуществу гражданина, подлежит возмещению в полном объеме лицом, причинившим вред."
    },
    {
        "situation": "Налоговая инспекция начислила штраф за несвоевременную подачу декларации по НДС.",
        "article": "Статья 119 НК РФ",
        "code": "НК РФ",
        "reasoning": "Непредставление налоговой декларации влечет взыскание штрафа."
    },
    {
        "situation": "Сосед курит на лестничной площадке, дым проникает в квартиру.",
        "article": "Статья 6.24 КоАП РФ",
        "code": "КоАП РФ",
        "reasoning": "Нарушение установленного федеральным законом запрета курения табака на отдельных территориях."
    }
]

def generate_mock_data(output_path: str, count: int = 50):
    data = []
    for _ in range(count):
        case = random.choice(MOCK_CASES).copy()
        # Add slight variation to make them technically "unique" objects in memory, 
        # though content is duplicated for this mock.
        data.append(case)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {count} mock cases to {output_path}")

if __name__ == "__main__":
    generate_mock_data("data/raw/synthetic_cases.json")
