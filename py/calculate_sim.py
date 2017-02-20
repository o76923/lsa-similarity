import string

from gensim.models import LsiModel
from gensim.corpora import Dictionary
import numpy as np
from os import getenv
from scipy.spatial.distance import cdist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import multiprocessing


class LSASim(object):
    PUNCTUATION_PATTERN = re.compile('[%s]' % re.escape(string.punctuation))

    def __init__(self, file_name, space_name):
        self.file_name = file_name
        self.space_name = space_name
        self.stemmer = PorterStemmer()
        self.dictionary = Dictionary.load("/app/data/%s.dictionary" % self.space_name)
        self.model = LsiModel.load("/app/data/%s.lsi" % self.space_name)
        self.alnum_patt = re.compile('[^a-z0-9]')
        self.stopwords = set(stopwords.words('english'))
        self.sentences = dict()
        self.vectors = dict()

    def load_sentences(self):
        with open("/app/data/%s.txt" % self.file_name) as in_file:
            in_file.readline()
            raw_sentences = {x[0]: x[1] for x in [l.split("\t") for l in in_file.readlines()]}
        self.sentences = raw_sentences

    def clean_sentence(self, sentence):
        try:
            return [self.stemmer.stem(w) for w in self.alnum_patt.sub(' ', sentence.lower()).split() if len(w) > 0 and w not in self.stopwords]
        except AttributeError:
            return []

    def process_sentence(self, item):
        k, v = item
        bow = self.dictionary.doc2bow(self.clean_sentence(v))
        ms = self.model[bow]
        return k, np.array([n[1] for n in ms])

    def calculate_similarities(self):
        print("starting sentence_dict", len(self.vectors))
        nulls = set(k for k in self.vectors if self.vectors[k] is None or len(self.vectors[k]) != 300)
        print("found nulls", len(nulls))
        with_data = list(self.vectors.keys() - nulls)
        print("reduced to with data", len(with_data))
        i = 0
        index_sentence_map = {}
        for k in with_data:
            index_sentence_map[i] = k
            i += 1
        print("made index_sentence_map", len(index_sentence_map))
        if i > 0:
            mat = np.array([self.vectors[k] for k in with_data])
            print("made mat")
            sims = cdist(mat, mat, 'cosine')
            print("calculated sims")
            sims = np.ones(shape=sims.shape, dtype=np.float) - sims
            print("subtracted sims from 1")
            return sims, index_sentence_map
        else:
            return []

    def trip_generator(self, sims):
        I, J = np.indices(sims.shape)
        triplets = np.column_stack(ar.ravel() for ar in (I, J, sims))
        for t in triplets:
            if t[0] <= t[1]:
                yield int(t[0]), int(t[1]), float(t[2])

    def main(self):
        self.load_sentences()
        print("loaded sentences")
        pool = multiprocessing.Pool(19)
        self.vectors = {k: v for k, v in (pool.map_async(func=self.process_sentence, iterable=self.sentences.items())).get()}
        pool.close()

        print("made sentence dict")
        sims, key_sentence_map = self.calculate_similarities()
        print("calculated similarities")

        with open("/app/data/%s_LSA_%s.csv" % (self.file_name, self.space_name), "w") as o:
            for l, r, s in self.trip_generator(sims):
                line = "%s,%s,%0.4f\n" % (key_sentence_map[l], key_sentence_map[r], s)
                o.write(line)
