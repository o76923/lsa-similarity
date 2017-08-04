import numpy as np


class Vectorizer(object):
    def __init__(self, dictionary, model, rot_mat=None):
        self.dictionary = dictionary
        self.model = model
        # self.s = np.diag(self.model.projection.s)
        self.rot_mat = rot_mat

    def vectorize(self, key, document):
        bow = self.dictionary.doc2bow(document)
        u = np.array([n[1] for n in self.model[bow]], dtype=np.float32)
        return key, u

    def rotated_vectorize(self, key, document):
        key, vector = self.vectorize(key, document)
        return key, np.dot(vector, self.rot_mat)

v: Vectorizer


def init_worker(d, m, r=None):
    global v
    v = Vectorizer(d, m, r)


def vectorize(key, document):
    global v
    return v.vectorize(key, document)


def rotated_vectorize(key, document):
    global v
    return v.vectorize(key, document)
