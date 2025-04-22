import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Константы
LEMMAS_FILE = 'lemmas.txt'
INVERTED_INDEX_FILE = 'inverted_index.txt'
TFIDF_DIR = 'lemmas_tf_idf/'


def load_lemmas() -> Dict[str, str]:
    """Загружает словарь лемматизации"""
    lemmas = {}
    try:
        with open(LEMMAS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                lemma = parts[0].rstrip(':')
                for word in parts[1:]:  # Пропускаем первую часть (лемму с :)
                    lemmas[word] = lemma
        return lemmas
    except FileNotFoundError:
        print(f"Ошибка: Файл {LEMMAS_FILE} не найден!")
        return {}


def load_inverted_index() -> Dict[str, List[int]]:
    """Загружает обратный индекс"""
    index = {}
    try:
        with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                index[data['word']] = data['inverted_array']
        return index
    except FileNotFoundError:
        print(f"Ошибка: Файл {INVERTED_INDEX_FILE} не найден!")
        return {}


def load_tfidf_data() -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Загружает TF-IDF и вычисляет длины векторов документов"""
    doc_tfidf = defaultdict(dict)
    doc_lengths = {}

    if not os.path.exists(TFIDF_DIR):
        print(f"Ошибка: Директория {TFIDF_DIR} не найдена!")
        return {}, {}

    for filename in os.listdir(TFIDF_DIR):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(TFIDF_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                lemma = parts[0]
                tfidf = float(parts[2])  # Берем tf-idf (третье значение)
                doc_tfidf[filename][lemma] = tfidf

        # Вычисляем длину вектора документа
        length = math.sqrt(sum(tfidf ** 2 for tfidf in doc_tfidf[filename].values()))
        doc_lengths[filename] = length if length != 0 else 1.0

    return dict(doc_tfidf), doc_lengths


def process_query(query: str, lemmas: Dict[str, str]) -> List[str]:
    """Обрабатывает поисковый запрос"""
    words = re.findall(r'\w+', query.lower())
    return [lemmas[word] for word in words if word in lemmas]


def search(
        query_lemmas: List[str],
        index: Dict[str, List[int]],
        doc_tfidf: Dict[str, Dict[str, float]],
        doc_lengths: Dict[str, float]
) -> List[Tuple[str, float]]:
    """Выполняет поиск и ранжирование документов"""
    # 1. Находим все релевантные документы
    relevant_docs = set()
    for lemma in query_lemmas:
        if lemma in index:
            for doc_id in index[lemma]:
                doc_name = f"page_{doc_id}.txt"
                relevant_docs.add(doc_name)

    # 2. Создаем вектор запроса (бинарный - 1 если термин есть в запросе)
    query_vector = {lemma: 1 for lemma in set(query_lemmas)}
    query_length = math.sqrt(len(query_vector))

    # 3. Вычисляем релевантность для каждого документа
    results = []
    for doc_name in relevant_docs:
        if doc_name not in doc_tfidf:
            continue

        # Скалярное произведение векторов запроса и документа
        dot_product = sum(
            query_vector.get(lemma, 0) * doc_tfidf[doc_name].get(lemma, 0)
            for lemma in query_vector
        )

        # Косинусная мера
        denominator = query_length * doc_lengths[doc_name]
        if denominator == 0:
            continue

        cosine_similarity = dot_product / denominator
        doc_id = doc_name.replace('page_', '').replace('.txt', '')
        results.append((doc_id, cosine_similarity))

    # Сортируем по убыванию релевантности
    return sorted(results, key=lambda x: x[1], reverse=True)


def main():
    """Основная функция поисковой системы"""
    print("Загрузка данных...")

    # Загрузка данных
    lemmas = load_lemmas()
    index = load_inverted_index()
    doc_tfidf, doc_lengths = load_tfidf_data()

    if not lemmas or not index or not doc_tfidf:
        print("Не удалось загрузить необходимые данные!")
        return

    # Проверка данных
    print("\nПроверка данных:")
    print(f"Загружено лемм: {len(lemmas)}")
    print(f"Загружено терминов в индексе: {len(index)}")
    print(f"Загружено документов с TF-IDF: {len(doc_tfidf)}")

    # Пример проверки конкретных терминов
    test_terms = ["мама", "добрый", "кошка"]
    for term in test_terms:
        lemma = lemmas.get(term, None)
        if lemma:
            print(f"\nТермин: '{term}' -> Лемма: '{lemma}'")
            print(f"Документы в индексе: {index.get(lemma, [])}")
            if lemma in index and index[lemma]:
                doc_id = index[lemma][0]
                doc_name = f"page_{doc_id}.txt"
                print(f"TF-IDF для документа {doc_name}: {doc_tfidf.get(doc_name, {}).get(lemma, 'не найден')}")

    print("\nПоисковая система готова к работе. Вводите запросы.")

    while True:
        query = input("\nПоисковый запрос (или 'q' для выхода): ").strip()
        if query.lower() == 'q':
            break

        if not query:
            continue

        # Обработка запроса
        query_lemmas = process_query(query, lemmas)
        if not query_lemmas:
            print("Не найдено подходящих лемм для запроса.")
            continue

        # Поиск и ранжирование
        results = search(query_lemmas, index, doc_tfidf, doc_lengths)

        # Вывод результатов
        if not results:
            print("Ничего не найдено.")
        else:
            print("\nРезультаты поиска (документ: релевантность):")
            for doc_id, score in results:
                print(f"Документ {doc_id}: {score:.6f}")


if __name__ == '__main__':
    main()