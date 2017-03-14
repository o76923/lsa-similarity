import re
from nltk import PorterStemmer
from py.configurator import SpaceSettings


class DocumentCleaner(object):
    def __init__(self, config: SpaceSettings):
        self.case_insensitive = config.case_insensitive
        self.remove_punctuation = config.remove_punctuation

        if config.remove_numbers:
            self.patt = re.compile('[^a-zA-Z]')
        else:
            self.patt = re.compile('[^a-zA-Z0-9]')
        if config.stopwords:
            self.stopwords = config.stopwords
        else:
            self.stopwords = None
        if config.stem:
            self.stem = PorterStemmer()
        else:
            self.stem = None

    def _try_stem(self, word):
        try:
            return self.stem.stem(word)
        except IndexError:
            return ""

    def clean_document(self, document):
        try:
            if self.case_insensitive:
                document = document.lower()
            if self.remove_punctuation:
                document = self.patt.sub(' ', document)
            words = document.split()
            if self.stopwords:
                words = [w for w in words if w not in self.stopwords]
            if self.stem:
                words = [self._try_stem(w) for w in words]
                if self.stopwords:
                    words = [w for w in words if w not in self.stopwords]
            words = [w for w in words if len(w) > 0]
            return words
        except AttributeError:
            return []

dc: DocumentCleaner


def init_worker(*args):
    global dc
    dc = DocumentCleaner(args[0])


def clean_document(document):
    global dc
    return dc.clean_document(document)


def clean_keyed_document(key, document):
    global dc
    return key, dc.clean_document(document)