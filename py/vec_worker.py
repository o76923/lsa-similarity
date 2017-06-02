import numpy as np


class Vectorizer(object):

    def __init__(self, dictionary, model):
        self.dictionary = dictionary
        self.model = model

    def vectorize(self, key, document):
        bow = self.dictionary.doc2bow(document)
        ms = self.model[bow]
        return key, np.array([n[1] for n in ms], dtype=np.float32)

v: Vectorizer


def init_worker(d, m):
    global v
    v = Vectorizer(d, m)


def vectorize(key, document):
    global v
    return v.vectorize(key, document)