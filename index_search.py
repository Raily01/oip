import os
from collections import defaultdict
from importlib import util

FILES_PATH = "downloaded_pages"
INVERTED_INDEX_PATH = "inverted_index.txt"


def module_from_file(module_name, file_path):
    spec = util.spec_from_file_location(module_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lemmatisation = module_from_file(
    "task_02", "task_02/tokens_and_lemmas_generator.py")

class IndexInverter:
    def __init__(self) -> None:
        self.inverted_index = defaultdict(list)

    def get_inverted_index(self):
        for root, _, files in os.walk(FILES_PATH):
            for index, file in enumerate(sorted(files), 1):
                lemmatisator = lemmatisation.Lemmatisator()
                lemmatisator.run_lemmatization(
                    lemmatisation.get_text_from_html(os.path.join(root, file))
                )
                for lemma in lemmatisator.lemmas.keys():
                    self.inverted_index[lemma].append(index)

    def write_inverted_index(self, path):
        with open(path, "w", encoding="UTF-8") as file:
            for word, inverted_array in self.inverted_index.items():
                file.write(
                    str(
                        {
                            "count": len(inverted_array),
                            "inverted_array": inverted_array,
                            "word": word,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    inverted_index = IndexInverter()
    inverted_index.get_inverted_index()
    inverted_index.write_inverted_index(INVERTED_INDEX_PATH)