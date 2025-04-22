from flask import Flask, render_template, request
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

app = Flask(__name__)

LEMMAS_FILE = 'lemmas.txt'
INVERTED_INDEX_FILE = 'inverted_index.txt'
TFIDF_DIR = 'lemmas_tf_idf/'


def load_lemmas() -> Dict[str, str]:
    lemmas = {}
    try:
        with open(LEMMAS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                lemma = parts[0].rstrip(':')
                for word in parts[1:]:
                    lemmas[word] = lemma
        return lemmas
    except FileNotFoundError:
        print(f"Ошибка: Файл {LEMMAS_FILE} не найден!")
        return {}


def load_inverted_index() -> Dict[str, List[int]]:
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
                tfidf = float(parts[2])
                doc_tfidf[filename][lemma] = tfidf

        length = math.sqrt(sum(tfidf ** 2 for tfidf in doc_tfidf[filename].values()))
        doc_lengths[filename] = length if length != 0 else 1.0

    return dict(doc_tfidf), doc_lengths


def process_query(query: str, lemmas: Dict[str, str]) -> List[str]:
    words = re.findall(r'\w+', query.lower())
    return [lemmas[word] for word in words if word in lemmas]


def search(
        query_lemmas: List[str],
        index: Dict[str, List[int]],
        doc_tfidf: Dict[str, Dict[str, float]],
        doc_lengths: Dict[str, float],
        top_n: int = 10
) -> List[Tuple[str, float]]:
    relevant_docs = set()
    for lemma in query_lemmas:
        if lemma in index:
            for doc_id in index[lemma]:
                doc_name = f"page_{doc_id}.txt"
                relevant_docs.add(doc_name)

    query_vector = {lemma: 1 for lemma in set(query_lemmas)}
    query_length = math.sqrt(len(query_vector))

    results = []
    for doc_name in relevant_docs:
        if doc_name not in doc_tfidf:
            continue

        dot_product = sum(
            query_vector.get(lemma, 0) * doc_tfidf[doc_name].get(lemma, 0)
            for lemma in query_vector
        )

        denominator = query_length * doc_lengths[doc_name]
        if denominator == 0:
            continue

        cosine_similarity = dot_product / denominator
        doc_id = doc_name.replace('page_', '').replace('.txt', '')
        results.append((doc_id, cosine_similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


lemmas = load_lemmas()
index = load_inverted_index()
doc_tfidf, doc_lengths = load_tfidf_data()


@app.route('/', methods=['GET', 'POST'])
def search_page():
    if request.method == 'POST':
        query = request.form['query']
        query_lemmas = process_query(query, lemmas)

        if not query_lemmas:
            return render_template('index.html', error="Не найдено подходящих лемм для запроса")

        results = search(query_lemmas, index, doc_tfidf, doc_lengths)

        if not results:
            return render_template('index.html', error="Ничего не найдено")

        return render_template('index.html', results=results, query=query)

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)

    # Создаем HTML шаблон
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Поисковая система</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-box {
            display: flex;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .results {
            margin-top: 20px;
        }
        .result-item {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .result-item:nth-child(odd) {
            background-color: #f9f9f9;
        }
        .score {
            color: #666;
            font-size: 14px;
        }
        .error {
            color: red;
            padding: 10px;
        }
    </style>
</head>
<body>
    <h1>Поисковая система</h1>
    <form method="POST">
        <div class="search-box">
            <input type="text" name="query" placeholder="Введите поисковый запрос..." value="{{ query if query else '' }}" required>
            <button type="submit">Найти</button>
        </div>
    </form>

    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}

    {% if results %}
    <div class="results">
        <h2>Результаты поиска (Топ-10):</h2>
        {% for doc_id, score in results %}
        <div class="result-item">
            Документ #{{ doc_id }} 
            <span class="score">(релевантность: {{ "%.4f"|format(score) }})</span>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>''')

    app.run(debug=True)

# каша библиотека огонь конец