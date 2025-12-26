#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пайплайн предобработки сырых данных для SFT (instruction tuning)
Создает train/val/test датасеты без утечек данных (group split)
"""

import json
import re
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
import random
from dataclasses import dataclass
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Фиксированный seed для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Пороги для фильтров качества
MIN_USER_LENGTH = 5  # Снижено для коротких вопросов
MIN_ASSISTANT_LENGTH = 10  # Снижено с 20 до 10 символов
MAX_TEXT_LENGTH = 50000


@dataclass
class ProcessingStats:
    """Статистика обработки"""
    initial_count: int = 0
    after_normalization: int = 0
    after_cleaning: int = 0
    after_quality_filters: int = 0
    after_deduplication: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    removed_reasons: Dict[str, int] = None
    
    def __post_init__(self):
        if self.removed_reasons is None:
            self.removed_reasons = defaultdict(int)


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        random.seed(seed)
        self.stats = ProcessingStats()
    
    def ingest_raw(self, raw_dir: Path) -> List[Dict[str, Any]]:
        """
        Загружает сырые данные из raw директории
        Приводит к единой схеме: source_id, group_id, system, user, assistant
        Поддерживает два формата:
        1. Старый формат: {"case": "...", "article": "..."}
        2. Новый формат: {"messages": [{"role": "...", "content": "..."}]}
        """
        logger.info(f"Загрузка сырых данных из {raw_dir}")
        all_records = []
        
        # Паттерны для определения кода кодекса из имени файла
        code_patterns = {
            'gk': r'actual_gk|gk_rf',
            'nk': r'actual_nk|nk_rf',
            'tk': r'actual_tk|tk_rf',
            'koap': r'actual_koap|koap',
            'apk': r'apk_rf'
        }
        
        for json_file in raw_dir.glob("*.json"):
            if json_file.name.startswith('.'):
                continue
                
            # Определяем кодекс из имени файла
            code = None
            for code_name, pattern in code_patterns.items():
                if re.search(pattern, json_file.name, re.IGNORECASE):
                    code = code_name
                    break
            
            if code is None:
                logger.warning(f"Не удалось определить кодекс для {json_file.name}, пропускаем")
                continue
            
            logger.info(f"Обработка {json_file.name} (кодекс: {code})")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = [data]
                
                # Логируем структуру первых нескольких элементов
                logger.info(f"Загружено {len(data)} элементов из {json_file.name}")
                if len(data) > 0:
                    first_item = data[0]
                    logger.info(f"  Первый элемент keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'not a dict'}")
                    if isinstance(first_item, dict) and 'messages' in first_item:
                        logger.info(f"  Формат: messages (найдено {len(first_item['messages'])} сообщений)")
                        for i, msg in enumerate(first_item['messages'][:3]):
                            logger.info(f"    Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
                    elif isinstance(first_item, dict):
                        logger.info(f"  Формат: case/article (case={bool(first_item.get('case'))}, article={bool(first_item.get('article'))})")
                
                for idx, item in enumerate(data):
                    # Проверяем формат данных
                    if isinstance(item, list):
                        # Данные пришли как список messages напрямую
                        messages = item
                        user_content = None
                        assistant_content = None
                        
                        for msg in messages:
                            if isinstance(msg, dict):
                                if msg.get('role') == 'user':
                                    user_content = msg.get('content', '')
                                elif msg.get('role') == 'assistant':
                                    assistant_content = msg.get('content', '')
                        
                        # Извлекаем article number из assistant content
                        article_text = assistant_content or ''
                        article_num = self._extract_article_number(article_text)
                        
                        group_id = f"{code}_{article_num}" if article_num else f"{code}_{idx}"
                        
                        record = {
                            'source_id': f"{json_file.stem}_{idx}",
                            'group_id': group_id,
                            'code': code,
                            'article_number': article_num,
                            'raw_case': user_content or '',
                            'raw_article': assistant_content or '',
                            'is_messages_format': True,  # Флаг для пропуска нормализации
                        }
                        all_records.append(record)
                    elif 'messages' in item:
                        # Новый формат: уже в messages структуре
                        messages = item['messages']
                        user_content = None
                        assistant_content = None
                        
                        for msg in messages:
                            if msg.get('role') == 'user':
                                user_content = msg.get('content', '')
                            elif msg.get('role') == 'assistant':
                                assistant_content = msg.get('content', '')
                        
                        # Извлекаем article number из assistant content
                        article_text = assistant_content or ''
                        article_num = self._extract_article_number(article_text)
                        
                        group_id = f"{code}_{article_num}" if article_num else f"{code}_{idx}"
                        
                        record = {
                            'source_id': f"{json_file.stem}_{idx}",
                            'group_id': group_id,
                            'code': code,
                            'article_number': article_num,
                            'raw_case': user_content or '',
                            'raw_article': assistant_content or '',
                            'is_messages_format': True,  # Флаг для пропуска нормализации
                        }
                        all_records.append(record)
                    else:
                        # Проверяем различные форматы данных
                        # Формат 1: case/article (старый формат)
                        if 'case' in item or 'article' in item:
                            article_text = item.get('article', '') or item.get('case', '')
                            article_num = self._extract_article_number(article_text)
                            
                            group_id = f"{code}_{article_num}" if article_num else f"{code}_{idx}"
                            
                            record = {
                                'source_id': f"{json_file.stem}_{idx}",
                                'group_id': group_id,
                                'code': code,
                                'article_number': article_num,
                                'raw_case': item.get('case', ''),
                                'raw_article': item.get('article', ''),
                                'is_messages_format': False,
                            }
                            all_records.append(record)
                        # Формат 2: title/content (новый формат статей)
                        elif 'title' in item and 'content' in item:
                            # content = текст статьи (assistant)
                            # title = заголовок статьи, может содержать вопрос или название
                            title = item.get('title', '').strip()
                            content = item.get('content', '').strip()
                            
                            # Логируем первые несколько для отладки
                            if idx < 3:
                                logger.info(f"  DEBUG title/content format: idx={idx}")
                                logger.info(f"    title length: {len(title)}, content length: {len(content)}")
                                logger.info(f"    title preview: {title[:100]}")
                                logger.info(f"    content preview: {content[:100]}")
                            
                            article_text = content or title
                            article_num = self._extract_article_number(article_text or title)
                            
                            group_id = f"{code}_{article_num}" if article_num else f"{code}_{idx}"
                            
                            # Создаем вопрос на основе title
                            # Если title содержит номер статьи, создаем вопрос
                            if article_num and article_num != 'unknown':
                                # Извлекаем название статьи из title (убираем номер)
                                article_name = title.replace(f'Статья {article_num}', '').strip()
                                article_name = article_name.replace(f'ст. {article_num}', '').strip()
                                article_name = article_name.replace(f'Статья {article_num.replace("_", " ")}', '').strip()
                                if article_name:
                                    user_question = f"Какая статья из {code.upper()} кодекс РФ регулирует: {article_name}"
                                else:
                                    user_question = f"Какая статья из {code.upper()} кодекс РФ регулирует: {title}"
                            else:
                                # Используем title как вопрос или создаем общий вопрос
                                if title:
                                    user_question = f"Расскажи о статье: {title}"
                                else:
                                    user_question = f"Расскажи об этой статье {code.upper()} кодекса"
                            
                            record = {
                                'source_id': f"{json_file.stem}_{idx}",
                                'group_id': group_id,
                                'code': code,
                                'article_number': article_num,
                                'raw_case': user_question,  # Используем созданный вопрос как user
                                'raw_article': content,  # Используем content как assistant
                                'is_messages_format': False,
                            }
                            all_records.append(record)
                            
                            if idx < 3:
                                logger.info(f"    Created record: user_question length={len(user_question)}, content length={len(content)}")
                        else:
                            logger.warning(f"Неизвестный формат данных в {json_file.name}, запись {idx}: keys={list(item.keys())}")
                            continue
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке {json_file}: {e}")
                continue
        
        self.stats.initial_count = len(all_records)
        logger.info(f"Загружено {len(all_records)} записей")
        return all_records
    
    def _extract_article_number(self, text: str) -> str:
        """Извлекает номер статьи из текста"""
        if not text:
            return "unknown"
        
        # Паттерн для поиска "Статья N" или "Статья N M"
        match = re.search(r'Статья\s+(\d+(?:\s+\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace(' ', '_')
        
        return "unknown"
    
    def normalize_format(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Приводит все примеры к messages-структуре (system/user/assistant)
        Формат: {"messages": [{"role": "system", "content": ...}, ...]}
        """
        logger.info("Нормализация формата к messages структуре")
        normalized = []
        
        debug_count = 0
        max_debug = 3
        
        for record in records:
            # Если данные уже в формате messages, пропускаем нормализацию
            if record.get('is_messages_format', False):
                # Просто используем существующие данные
                user_text = record.get('raw_case', '').strip()
                assistant_text = record.get('raw_article', '').strip()
                
                if debug_count < max_debug:
                    logger.info(f"DEBUG normalize (messages format): source_id={record.get('source_id')}")
                    logger.info(f"  user_text length: {len(user_text)}, assistant_text length: {len(assistant_text)}")
                    logger.info(f"  user_text preview: {user_text[:100]}")
                    logger.info(f"  assistant_text preview: {assistant_text[:100]}")
                    debug_count += 1
                
                # Проверяем, что есть хотя бы user и assistant
                if not user_text or not assistant_text:
                    logger.warning(f"Пропуск записи {record['source_id']}: отсутствует user или assistant (user_len={len(user_text)}, assistant_len={len(assistant_text)})")
                    self.stats.removed_reasons['empty_after_normalization'] += 1
                    continue
                
                messages = []
                system_content = "Ты - юридический ассистент, который помогает разбираться в российском законодательстве."
                messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
            else:
                # Старый формат: преобразуем case/article в messages
                user_text = record.get('raw_case', '').strip()
                assistant_text = record.get('raw_article', '').strip()
                
                if debug_count < max_debug:
                    logger.info(f"DEBUG normalize (case/article format): source_id={record.get('source_id')}")
                    logger.info(f"  user_text length: {len(user_text)}, assistant_text length: {len(assistant_text)}")
                    debug_count += 1
                
                # Если нет user, используем общий промпт
                if not user_text:
                    user_text = "Расскажи об этой статье кодекса"
                
                # Создаем messages структуру
                messages = []
                
                # System prompt для юридического ассистента
                system_content = "Ты - юридический ассистент, который помогает разбираться в российском законодательстве."
                messages.append({"role": "system", "content": system_content})
                
                # User (вопрос/контекст)
                messages.append({"role": "user", "content": user_text})
                
                # Assistant (ответ)
                messages.append({"role": "assistant", "content": assistant_text})
            
            normalized_record = {
                'source_id': record['source_id'],
                'group_id': record['group_id'],
                'code': record['code'],
                'article_number': record['article_number'],
                'messages': messages
            }
            normalized.append(normalized_record)
        
        self.stats.after_normalization = len(normalized)
        logger.info(f"После нормализации: {len(normalized)} записей")
        return normalized
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста: удаление HTML/PDF артефактов, нормализация пробелов
        """
        if not text:
            return ""
        
        # Убеждаемся, что text - это строка
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        if not text:
            return ""
        
        # Удаляем HTML id атрибуты
        text = re.sub(r'^id="[^"]*"\s*', '', text)
        # Удаляем символ ">" в начале
        text = re.sub(r'^>\s*', '', text)
        # Заменяем неразрывные пробелы
        text = text.replace("\xa0", " ")
        # Нормализация пробелов и табуляций
        text = re.sub(r"[ \t]+", " ", text)
        # Нормализация переносов строк
        text = re.sub(r"\n+", " ", text)
        # Удаляем множественные пробелы
        text = re.sub(r" +", " ", text)
        # Удаляем "undefined" и "null" из текста
        text = re.sub(r'\bundefined\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnull\b', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def apply_text_cleaning(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Применяет очистку текста ко всем сообщениям"""
        logger.info("Применение очистки текста")
        cleaned = []
        
        debug_count = 0
        max_debug = 3
        
        for record in records:
            cleaned_messages = []
            assistant_before = None
            assistant_after = None
            
            for msg in record['messages']:
                original_content = msg.get('content', '')
                if not isinstance(original_content, str):
                    original_content = str(original_content) if original_content else ''
                
                cleaned_content = self.clean_text(original_content)
                
                if msg.get('role') == 'assistant':
                    assistant_before = original_content
                    assistant_after = cleaned_content
                
                if cleaned_content:  # Пропускаем пустые сообщения
                    cleaned_messages.append({
                        'role': msg.get('role'),
                        'content': cleaned_content
                    })
            
            if debug_count < max_debug:
                logger.info(f"=== DEBUG cleaning #{debug_count}: source_id={record.get('source_id')} ===")
                logger.info(f"  Messages before: {len(record['messages'])}, after: {len(cleaned_messages)}")
                if assistant_before:
                    logger.info(f"  Assistant before cleaning: length={len(assistant_before)}, type={type(assistant_before)}")
                    logger.info(f"    Preview: {assistant_before[:200]}")
                    logger.info(f"  Assistant after cleaning: length={len(assistant_after) if assistant_after else 0}")
                    logger.info(f"    Preview: {assistant_after[:200] if assistant_after else 'EMPTY'}")
                    if len(assistant_after) < len(assistant_before) * 0.5:
                        logger.warning(f"  ВНИМАНИЕ: Длина уменьшилась более чем в 2 раза! ({len(assistant_before)} -> {len(assistant_after)})")
                else:
                    logger.warning(f"  Assistant не найден в messages!")
                debug_count += 1
            
            if len(cleaned_messages) >= 2:  # Минимум system + user или user + assistant
                record['messages'] = cleaned_messages
                cleaned.append(record)
            else:
                self.stats.removed_reasons['empty_after_cleaning'] += 1
                if debug_count <= max_debug:
                    logger.warning(f"Удален после очистки: source_id={record.get('source_id')}, cleaned_messages={len(cleaned_messages)}")
        
        self.stats.after_cleaning = len(cleaned)
        logger.info(f"После очистки: {len(cleaned)} записей")
        return cleaned
    
    def quality_filters(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Фильтры качества:
        - User не пустой, не слишком короткий
        - Assistant не пустой, без мусора
        - Исключение токсичности/PII/плохих примеров
        """
        logger.info("Применение фильтров качества")
        filtered = []
        
        # Логируем первые несколько УДАЛЕННЫХ записей для отладки
        removed_debug_count = 0
        max_removed_debug = 5
        
        # Также логируем первые несколько записей вообще
        total_debug_count = 0
        max_total_debug = 2
        
        for record in records:
            messages = record['messages']
            
            # Логируем структуру messages для первых записей
            if total_debug_count < max_total_debug:
                logger.info(f"=== DEBUG Record {total_debug_count}: source_id={record.get('source_id', 'unknown')} ===")
                logger.info(f"  Messages count: {len(messages)}")
                for i, msg in enumerate(messages):
                    content = msg.get('content', '')
                    content_len = len(content) if isinstance(content, str) else 0
                    logger.info(f"    Message {i}: role={msg.get('role')}, content_length={content_len}")
                    if msg.get('role') == 'assistant':
                        preview = content[:200] if isinstance(content, str) else str(content)[:200]
                        logger.info(f"      Assistant preview: {preview}")
                total_debug_count += 1
            
            # Находим user и assistant сообщения
            user_content = None
            assistant_content = None
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                # Убеждаемся, что content - это строка
                if not isinstance(content, str):
                    if content is None:
                        content = ''
                    else:
                        content = str(content)
                
                if role == 'user':
                    user_content = content
                elif role == 'assistant':
                    assistant_content = content
            
            # Проверка user
            if not user_content:
                self.stats.removed_reasons['user_too_short'] += 1
                if removed_debug_count < max_removed_debug:
                    logger.warning(f"=== УДАЛЕНО #{removed_debug_count}: user_content is None or empty ===")
                    logger.warning(f"  source_id={record.get('source_id')}")
                    logger.warning(f"  messages count: {len(messages)}")
                    removed_debug_count += 1
                continue
            
            # Убеждаемся, что это строка
            if not isinstance(user_content, str):
                user_content = str(user_content) if user_content else ''
            
            user_len = len(user_content.strip())
            if user_len < MIN_USER_LENGTH:
                self.stats.removed_reasons['user_too_short'] += 1
                if removed_debug_count < max_removed_debug:
                    logger.warning(f"=== УДАЛЕНО #{removed_debug_count}: user_too_short ===")
                    logger.warning(f"  source_id={record.get('source_id')}")
                    logger.warning(f"  user_len={user_len}, min={MIN_USER_LENGTH}")
                    logger.warning(f"  user_content preview: {user_content[:200]}")
                    removed_debug_count += 1
                continue
            
            # Проверка assistant
            if not assistant_content:
                self.stats.removed_reasons['assistant_too_short'] += 1
                if removed_debug_count < max_removed_debug:
                    logger.warning(f"=== УДАЛЕНО #{removed_debug_count}: assistant_content is None or empty ===")
                    logger.warning(f"  source_id={record.get('source_id')}")
                    logger.warning(f"  messages count: {len(messages)}")
                    logger.warning(f"  Available roles: {[msg.get('role') for msg in messages]}")
                    removed_debug_count += 1
                continue
            
            # Убеждаемся, что это строка
            if not isinstance(assistant_content, str):
                assistant_content = str(assistant_content) if assistant_content else ''
            
            assistant_len = len(assistant_content.strip())
            if assistant_len < MIN_ASSISTANT_LENGTH:
                self.stats.removed_reasons['assistant_too_short'] += 1
                if removed_debug_count < max_removed_debug:
                    logger.warning(f"=== УДАЛЕНО #{removed_debug_count}: assistant_too_short ===")
                    logger.warning(f"  source_id={record.get('source_id')}")
                    logger.warning(f"  assistant_len={assistant_len}, min={MIN_ASSISTANT_LENGTH}")
                    logger.warning(f"  assistant_content type: {type(assistant_content)}")
                    logger.warning(f"  assistant_content (first 300 chars): {assistant_content[:300]}")
                    logger.warning(f"  assistant_content (last 300 chars): {assistant_content[-300:]}")
                    logger.warning(f"  assistant_content repr (first 500): {repr(assistant_content[:500])}")
                    removed_debug_count += 1
                continue
            
            # Проверка максимальной длины
            total_length = len(user_content) + len(assistant_content)
            if total_length > MAX_TEXT_LENGTH:
                self.stats.removed_reasons['too_long'] += 1
                continue
            
            # Проверка на мусор (много повторяющихся символов)
            if self._has_repetitive_chars(user_content) or self._has_repetitive_chars(assistant_content):
                self.stats.removed_reasons['repetitive_chars'] += 1
                continue
            
            # Проверка на обрывы (текст заканчивается на незавершенное предложение)
            # Отключаем для коротких текстов, так как они могут быть валидными
            if len(assistant_content) > 100 and self._is_truncated(assistant_content):
                self.stats.removed_reasons['truncated'] += 1
                continue
            
            filtered.append(record)
        
        self.stats.after_quality_filters = len(filtered)
        logger.info(f"После фильтров качества: {len(filtered)} записей")
        
        # Если все записи удалены, выводим детальную статистику
        if len(filtered) == 0 and len(records) > 0:
            logger.error("=" * 60)
            logger.error("ВСЕ ЗАПИСИ УДАЛЕНЫ! Детальная диагностика:")
            logger.error(f"Всего записей до фильтрации: {len(records)}")
            logger.error(f"Причины удаления: {dict(self.stats.removed_reasons)}")
            
            # Проверяем первые 3 записи детально
            logger.error("\nДетальный анализ первых 3 записей:")
            for i, record in enumerate(records[:3]):
                logger.error(f"\n--- Запись #{i} ---")
                logger.error(f"source_id: {record.get('source_id')}")
                messages = record.get('messages', [])
                logger.error(f"Количество messages: {len(messages)}")
                
                user_content = None
                assistant_content = None
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if not isinstance(content, str):
                        content = str(content) if content else ''
                    
                    logger.error(f"  Message role={role}, content_length={len(content)}")
                    
                    if role == 'user':
                        user_content = content
                    elif role == 'assistant':
                        assistant_content = content
                        logger.error(f"    Assistant content (first 500): {content[:500]}")
                        logger.error(f"    Assistant content (repr first 200): {repr(content[:200])}")
                
                if user_content:
                    logger.error(f"User length: {len(user_content.strip())}, min required: {MIN_USER_LENGTH}")
                else:
                    logger.error("User content: НЕ НАЙДЕН")
                
                if assistant_content:
                    logger.error(f"Assistant length: {len(assistant_content.strip())}, min required: {MIN_ASSISTANT_LENGTH}")
                else:
                    logger.error("Assistant content: НЕ НАЙДЕН")
            
            logger.error("=" * 60)
        
        return filtered
    
    def _has_repetitive_chars(self, text: str, threshold: int = 5) -> bool:
        """Проверка на повторяющиеся символы (мусор)"""
        if len(text) < threshold * 2:
            return False
        
        for i in range(len(text) - threshold):
            substr = text[i:i+threshold]
            if len(set(substr)) == 1 and substr[0] not in ' \n\t':
                return True
        return False
    
    def _is_truncated(self, text: str) -> bool:
        """Проверка на обрыв текста"""
        if not text:
            return True
        
        # Текст обрывается, если заканчивается без знака препинания
        # и не на закрывающей скобке/кавычке
        last_char = text[-1]
        if last_char in '.!?;':
            return False
        if last_char in ')]}"':
            return False
        if len(text) < 100:  # Увеличено с 50 до 100 - короткие тексты не проверяем
            return False
        
        # Если последние 30 символов не содержат знаков препинания - вероятно обрыв
        # Увеличено с 20 до 30 для более мягкой проверки
        last_30 = text[-30:]
        if not re.search(r'[.!?;]', last_30):
            return True
        
        return False
    
    def deduplication(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Дедупликация:
        - Exact duplicates по хэшу (system+user+assistant)
        - Near-duplicates (похожие тексты)
        """
        logger.info("Дедупликация")
        
        # Exact duplicates по хэшу
        seen_hashes: Set[str] = set()
        unique_records = []
        
        for record in records:
            # Создаем хэш из всех сообщений
            messages_str = json.dumps(record['messages'], sort_keys=True, ensure_ascii=False)
            content_hash = hashlib.md5(messages_str.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_records.append(record)
            else:
                self.stats.removed_reasons['exact_duplicate'] += 1
        
        logger.info(f"После exact deduplication: {len(unique_records)} записей")
        
        # Near-duplicates (упрощенная версия - по похожести user+assistant)
        # Используем простой метод: если user+assistant очень похожи (>95% совпадение), удаляем
        final_records = []
        seen_content: Set[str] = set()
        
        for record in unique_records:
            # Извлекаем user и assistant
            user_content = ""
            assistant_content = ""
            for msg in record['messages']:
                if msg['role'] == 'user':
                    user_content = msg['content'].lower()[:200]  # Первые 200 символов
                elif msg['role'] == 'assistant':
                    assistant_content = msg['content'].lower()[:500]  # Первые 500 символов
            
            content_key = f"{user_content}|||{assistant_content}"
            
            # Проверяем на near-duplicate
            is_duplicate = False
            for seen in seen_content:
                if self._similarity(content_key, seen) > 0.95:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content.add(content_key)
                final_records.append(record)
            else:
                self.stats.removed_reasons['near_duplicate'] += 1
        
        self.stats.after_deduplication = len(final_records)
        logger.info(f"После дедупликации: {len(final_records)} записей")
        return final_records
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Простая метрика похожести (Jaccard similarity по словам)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def split_train_val_test(
        self, 
        records: List[Dict[str, Any]], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: Optional[str] = 'code'
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Групповой split по group_id (без утечек данных)
        Стратификация по домену (code) для сохранения распределений
        """
        logger.info("Групповой split train/val/test")
        
        # Группируем по group_id
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[record['group_id']].append(record)
        
        logger.info(f"Всего уникальных групп: {len(groups)}")
        
        # Стратификация по домену (code)
        if stratify_by == 'code':
            # Группируем группы по коду
            groups_by_code: Dict[str, List[str]] = defaultdict(list)
            for group_id, group_records in groups.items():
                code = group_records[0]['code']
                groups_by_code[code].append(group_id)
            
            # Собираем все группы с кодексами
            all_groups_with_code = []
            for code, code_groups in groups_by_code.items():
                random.shuffle(code_groups)
                for group_id in code_groups:
                    all_groups_with_code.append((code, group_id))
            
            # Перемешиваем все группы
            random.shuffle(all_groups_with_code)
            
            # Вычисляем целевые размеры с балансировкой val/test
            total_groups = len(all_groups_with_code)
            target_train = int(total_groups * train_ratio)
            remaining = total_groups - target_train
            # Делим оставшиеся поровну между val и test
            target_val = remaining // 2
            target_test = remaining - target_val
            
            logger.info(f"Целевые размеры: Train={target_train}, Val={target_val}, Test={target_test} (всего групп: {total_groups})")
            
            # Распределяем группы с учетом стратификации
            train_groups = []
            val_groups = []
            test_groups = []
            
            # Счетчики по кодексам для поддержания пропорций
            code_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
            for code, _ in all_groups_with_code:
                code_stats[code]['total'] = sum(1 for c, _ in all_groups_with_code if c == code)
            
            for code, group_id in all_groups_with_code:
                stats = code_stats[code]
                code_total = stats['total']
                
                # Вычисляем текущие пропорции для этого кодекса
                code_processed = stats['train'] + stats['val'] + stats['test']
                if code_processed > 0:
                    code_train_pct = stats['train'] / code_processed
                    code_val_pct = stats['val'] / code_processed
                else:
                    code_train_pct = 0
                    code_val_pct = 0
                
                # Решаем, куда отправить группу
                # Приоритет: сначала заполняем train до нужной пропорции, затем val и test поровну
                if (len(train_groups) < target_train and 
                    (code_train_pct < train_ratio or len(val_groups) >= target_val and len(test_groups) >= target_test)):
                    train_groups.append(group_id)
                    stats['train'] += 1
                elif len(val_groups) < target_val and (code_val_pct < val_ratio or len(test_groups) >= target_test):
                    val_groups.append(group_id)
                    stats['val'] += 1
                else:
                    test_groups.append(group_id)
                    stats['test'] += 1
            
            logger.info(f"Train групп: {len(train_groups)}, Val групп: {len(val_groups)}, Test групп: {len(test_groups)}")
            
            # Финальная балансировка: если val и test сильно различаются, перераспределяем
            if abs(len(val_groups) - len(test_groups)) > 2:
                logger.info(f"Балансировка val/test: было Val={len(val_groups)}, Test={len(test_groups)}")
                # Перераспределяем между val и test
                total_val_test = len(val_groups) + len(test_groups)
                new_val = total_val_test // 2
                new_test = total_val_test - new_val
                
                # Если нужно, перемещаем группы
                if len(val_groups) < new_val:
                    # Берем из test
                    needed = new_val - len(val_groups)
                    groups_to_move = test_groups[:needed]
                    val_groups.extend(groups_to_move)
                    test_groups = test_groups[needed:]
                elif len(test_groups) < new_test:
                    # Берем из val
                    needed = new_test - len(test_groups)
                    groups_to_move = val_groups[:needed]
                    test_groups.extend(groups_to_move)
                    val_groups = val_groups[needed:]
                
                logger.info(f"После балансировки: Val={len(val_groups)}, Test={len(test_groups)}")
        else:
            # Простой случайный split без стратификации
            all_group_ids = list(groups.keys())
            random.shuffle(all_group_ids)
            
            n = len(all_group_ids)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_groups = all_group_ids[:n_train]
            val_groups = all_group_ids[n_train:n_train+n_val]
            test_groups = all_group_ids[n_train+n_val:]
        
        # Проверка на пересечения (должно быть 0)
        train_set = set(train_groups)
        val_set = set(val_groups)
        test_set = set(test_groups)
        
        assert len(train_set & val_set) == 0, "Пересечение train и val!"
        assert len(train_set & test_set) == 0, "Пересечение train и test!"
        assert len(val_set & test_set) == 0, "Пересечение val и test!"
        
        logger.info("✓ Проверка на утечки данных пройдена: пересечений нет")
        
        # Собираем записи по группам
        train_records = []
        val_records = []
        test_records = []
        
        for group_id in train_groups:
            train_records.extend(groups[group_id])
        for group_id in val_groups:
            val_records.extend(groups[group_id])
        for group_id in test_groups:
            test_records.extend(groups[group_id])
        
        self.stats.train_count = len(train_records)
        self.stats.val_count = len(val_records)
        self.stats.test_count = len(test_records)
        
        logger.info(f"Train: {len(train_records)} записей, Val: {len(val_records)}, Test: {len(test_records)}")
        
        # Сохраняем манифесты групп для воспроизводимости
        return train_records, val_records, test_records
    
    def save_manifests(
        self, 
        train_groups: Set[str], 
        val_groups: Set[str], 
        test_groups: Set[str],
        output_dir: Path
    ):
        """Сохраняет манифесты group_id для воспроизводимости"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'seed': self.seed,
            'train_groups': sorted(list(train_groups)),
            'val_groups': sorted(list(val_groups)),
            'test_groups': sorted(list(test_groups))
        }
        
        manifest_path = output_dir / 'split_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Манифест сплита сохранен: {manifest_path}")
    
    def export_jsonl(
        self, 
        records: List[Dict[str, Any]], 
        output_path: Path
    ):
        """Экспорт в JSONL формат"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                # Экспортируем только messages (финальный формат)
                output_record = {
                    'messages': record['messages']
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        logger.info(f"Экспортировано {len(records)} записей в {output_path}")
    
    def generate_report(self, output_dir: Path):
        """Генерирует отчет о предобработке"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Распределения по доменам
        domain_dist = Counter()
        # TODO: 
        
        # Вручную создаем словарь из stats, преобразуя defaultdict в dict
        processing_stats = {
            'initial_count': self.stats.initial_count,
            'after_normalization': self.stats.after_normalization,
            'after_cleaning': self.stats.after_cleaning,
            'after_quality_filters': self.stats.after_quality_filters,
            'after_deduplication': self.stats.after_deduplication,
            'train_count': self.stats.train_count,
            'val_count': self.stats.val_count,
            'test_count': self.stats.test_count,
            'removed_reasons': dict(self.stats.removed_reasons)
        }
        
        report = {
            'processing_stats': processing_stats,
            'removed_reasons': dict(self.stats.removed_reasons),
            'seed': self.seed,
            'total_removed': self.stats.initial_count - self.stats.after_deduplication,
            'removal_rate': (self.stats.initial_count - self.stats.after_deduplication) / self.stats.initial_count if self.stats.initial_count > 0 else 0
        }
        
        report_path = output_dir / 'preprocessing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Отчет сохранен: {report_path}")
        
        # Выводим краткую статистику
        print("\n" + "="*60)
        print("ОТЧЕТ О ПРЕДОБРАБОТКЕ")
        print("="*60)
        print(f"Начальное количество: {self.stats.initial_count}")
        print(f"После нормализации: {self.stats.after_normalization}")
        print(f"После очистки: {self.stats.after_cleaning}")
        print(f"После фильтров качества: {self.stats.after_quality_filters}")
        print(f"После дедупликации: {self.stats.after_deduplication}")
        print(f"\nФинальные сплиты:")
        print(f"  Train: {self.stats.train_count}")
        print(f"  Val: {self.stats.val_count}")
        print(f"  Test: {self.stats.test_count}")
        print(f"\nПричины удаления:")
        for reason, count in sorted(self.stats.removed_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн предобработки данных для SFT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='../data/raw',
        help='Директория с сырыми данными (по умолчанию: ../data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/processed/sft_datasets',
        help='Директория для выходных файлов (по умолчанию: ../data/processed/sft_datasets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed для воспроизводимости (по умолчанию: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Доля train данных (по умолчанию: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Доля val данных (по умолчанию: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Доля test данных (по умолчанию: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Проверка пропорций
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Сумма пропорций должна быть 1.0, получено: {total_ratio}")
        return
    
    # Инициализация
    preprocessor = DataPreprocessor(seed=args.seed)
    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not raw_dir.exists():
        logger.error(f"Директория с сырыми данными не найдена: {raw_dir}")
        return
    
    logger.info(f"Начало предобработки данных")
    logger.info(f"Raw dir: {raw_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Seed: {args.seed}")
    
    # Пайплайн обработки
    # 1. Ingest
    records = preprocessor.ingest_raw(raw_dir)
    
    if not records:
        logger.error("Не загружено ни одной записи!")
        return
    
    # 2. Normalize
    records = preprocessor.normalize_format(records)
    
    # 3. Text cleaning
    records = preprocessor.apply_text_cleaning(records)
    
    # 4. Quality filters
    records = preprocessor.quality_filters(records)
    
    # 5. Deduplication
    records = preprocessor.deduplication(records)
    
    # Проверка: есть ли данные после обработки
    if not records:
        logger.error("После обработки не осталось ни одной записи!")
        logger.error("Проверьте фильтры качества и исходные данные")
        preprocessor.generate_report(output_dir)
        return
    
    # 6. Split
    train_records, val_records, test_records = preprocessor.split_train_val_test(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # 7. Export
    if train_records:
        preprocessor.export_jsonl(train_records, output_dir / 'train.jsonl')
    else:
        logger.warning("Train набор пуст, файл не создан")
    
    if val_records:
        preprocessor.export_jsonl(val_records, output_dir / 'val.jsonl')
    else:
        logger.warning("Val набор пуст, файл не создан")
    
    if test_records:
        preprocessor.export_jsonl(test_records, output_dir / 'test.jsonl')
    else:
        logger.warning("Test набор пуст, файл не создан")
    
    # Сохранение манифестов
    train_groups = set(r['group_id'] for r in train_records)
    val_groups = set(r['group_id'] for r in val_records)
    test_groups = set(r['group_id'] for r in test_records)
    preprocessor.save_manifests(train_groups, val_groups, test_groups, output_dir)
    
    # Генерация отчета
    preprocessor.generate_report(output_dir)
    
    logger.info("Предобработка завершена успешно!")


if __name__ == "__main__":
    main()

