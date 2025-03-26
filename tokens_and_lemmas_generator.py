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
        self.tokens = set()
        self.lemmas = defaultdict(set)

    def run_lemmatization(self, text):
        self.tokens.update(self.tokenizer.tokenize(text))
        self.filter_tokens()

    def filter_tokens(self):
        bad_tokens = set()
        for token in self.tokens:
            morph = self.morph_analyzer.parse(token)
            if (
                    any([x for x in self.BAD_TOKENS_TAGS if x in morph[0].tag])
                    or token in self.stop_words
            ):
                bad_tokens.add(token)
                continue
            if morph[0].score >= 0.5:
                self.lemmas[morph[0].normal_form].add(token)
        self.tokens = self.tokens - bad_tokens

    def write_tokens(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.tokens))

    def write_lemmas(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for token, lemmas in self.lemmas.items():
                f.write(f"{token} {' '.join(lemmas)}\n")


if __name__ == "__main__":
    lemmatisator = Lemmatisator()
    pages_texts = []
    directory = 'downloaded_pages'
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            pages_texts.append(get_text_from_html(file_path))

    lemmatisator.run_lemmatization(" ".join(pages_texts))
    lemmatisator.write_tokens("tokens.txt")
    lemmatisator.write_lemmas("lemmas.txt")