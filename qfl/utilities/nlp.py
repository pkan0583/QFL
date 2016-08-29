from collections import defaultdict
from gensim import corpora, models, similarities


class DocumentsAnalyzer(object):

    @staticmethod
    def analyze_documents(documents=None, common_words=None):

        texts = DocumentsAnalyzer.remove_common_words(documents, common_words)
        texts = DocumentsAnalyzer.remove_solitary_words(texts)
        dictionary = corpora.Dictionary(texts)

    @staticmethod
    def get_default_common_words():
        common_words = set('for a of the and to in -'.split())
        return common_words

    @staticmethod
    def remove_common_words(documents=None, common_words=None):

        if common_words is None:
            common_words = DocumentsAnalyzer.get_default_common_words()

        texts = [[word for word in document.lower().split()
                  if word not in common_words]
                 for document in documents]

        return texts

    @staticmethod
    def remove_solitary_words(texts=None):

        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] > 1]
                 for text in texts]

        return texts


