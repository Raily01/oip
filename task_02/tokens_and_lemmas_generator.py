import os
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pymorphy2
import nltk

# Download the stopwords dataset (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def get_text_from_html(file_path):
    with open(file_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    return " ".join(soup.stripped_strings)


class Lemmatisator:
    BAD_TOKENS_TAGS = {"PREP", "CONJ", "PRCL", "INTJ", "LATN", "PNCT", "NUMB", "ROMN", "UNKN"}

    def __init__(self):
        self.stop_words = set(stopwords.words("russian"))
        self.tokenizer = WordPunctTokenizer()
        self.morph_analyzer = pymorphy2.MorphAnalyzer()
        self.tokens = set()  # Для хранения токенов
        self.lemmas = defaultdict(set)  # Для хранения лемм


    def run_lemmatization(self, text):
        tokens = set(self.tokenizer.tokenize(text))
        filtered_tokens = self.filter_tokens(tokens)
        lemmas = self.build_lemmas(filtered_tokens)
        return filtered_tokens, lemmas

    def filter_tokens(self, tokens):
        good_tokens = set()
        for token in tokens:
            morph = self.morph_analyzer.parse(token)
            if (
                    any([x for x in self.BAD_TOKENS_TAGS if x in morph[0].tag])
                    or token in self.stop_words
            ):
                continue
            if morph[0].score >= 0.5:
                good_tokens.add(token)
        self.tokens = good_tokens  # Сохраняем токены в атрибуте экземпляра

        return good_tokens

    def build_lemmas(self, tokens):
        lemmas = defaultdict(set)
        for token in tokens:
            morph = self.morph_analyzer.parse(token)
            lemma = morph[0].normal_form
            lemmas[lemma].add(token)
            self.lemmas[lemma].add(token)  # Сохраняем леммы в атрибуте экземпляра
        return lemmas


def write_tokens(tokens, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))


def write_lemmas(lemmas, path):
    with open(path, "w", encoding="utf-8") as f:
        for lemma, tokens in lemmas.items():
            f.write(f"{lemma} {' '.join(tokens)}\n")


if __name__ == "__main__":
    lemmatisator = Lemmatisator()
    directory = 'downloaded_pages'
    output_directory = 'output'

    # Создаем директорию для вывода, если она не существует
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            text = get_text_from_html(file_path)

            # Запускаем лемматизацию для текста текущей страницы
            tokens, lemmas = lemmatisator.run_lemmatization(text)

            # Формируем пути для сохранения результатов
            base_name = os.path.splitext(filename)[0]
            tokens_file = os.path.join(output_directory, f"{base_name}_tokens.txt")
            lemmas_file = os.path.join(output_directory, f"{base_name}_lemmas.txt")

            # Сохраняем токены и леммы в файлы
            write_tokens(tokens, tokens_file)
            write_lemmas(lemmas, lemmas_file)