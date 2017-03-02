import string
from itertools import product, combinations_with_replacement, starmap
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import numpy as np
from scipy.spatial.distance import cdist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import multiprocessing as mp
from os import getenv

TARGET = getenv("TARGET")
SPACE = getenv("SPACE")


def write_sims(sims, sl_a, sl_b):
    lines = []
    sim_iterator = sims.flat
    for l in sl_a:
        for r in sl_b:
            sim = sim_iterator.__next__()
            lines.append("{},{},{:0.4f}\n".format(l, r, sim))
    fn = "/app/data/%s_LSA_%s.csv" % (TARGET, SPACE)
    with open(fn, "a") as o:
        o.write("".join(lines))


def matrix_sim(data_a, data_b):
    mat_a, sl_a = data_a
    mat_b, sl_b = data_b
    sims = cdist(mat_a, mat_b, 'cosine')
    sims = np.ones(shape=sims.shape, dtype=np.float) - sims
    write_sims(sims, sl_a, sl_b)


class LSASim(object):
    PUNCTUATION_PATTERN = re.compile('[%s]' % re.escape(string.punctuation))

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.dictionary = Dictionary.load("/app/data/%s.dictionary" % SPACE)
        self.model = LsiModel.load("/app/data/%s.lsi" % SPACE)
        self.alnum_patt = re.compile('[^a-z0-9]')
        self.stopwords = set(stopwords.words('english'))
        self.sentences = dict()
        self.vectors = dict()

    def load_sentences(self):
        with open("/app/data/%s.txt" % TARGET, encoding="utf-8-sig") as in_file:
            lines = in_file.readlines()
            try:
                raw_sentences = {int(x[0]): x[1] for x in [l.split("\t") for l in lines]}
            except ValueError:
                raw_sentences = {int(x[0]): x[1] for x in [l.split("\t") for l in lines[1:]]}
        self.sentences = raw_sentences

    def _try_stem(self, word):
        try:
            return self.stemmer.stem(word)
        except IndexError:
            return ""

    def clean_sentence(self, sentence):
        try:
            return [w for w in [self._try_stem(w) for w in self.alnum_patt.sub(' ', sentence.lower()).split() if
                                w not in self.stopwords] if len(w) > 0]
        except AttributeError:
            return []

    def process_sentence(self, item):
        k, v = item
        bow = self.dictionary.doc2bow(self.clean_sentence(v))
        ms = self.model[bow]
        return k, np.array([n[1] for n in ms])

    def mini_mats(self, with_data, n):
        for i in range(0, len(with_data), n):
            sl = [k for k in with_data[i:i + n]]
            mat = np.array([self.vectors[k] for k in sl])
            yield mat, sl

    def filter_nulls(self):
        nulls = set(k for k in self.vectors if self.vectors[k] is None or len(self.vectors[k]) != 300)
        with_data = sorted(list(self.vectors.keys() - nulls))
        print("removed vectorless sentences", len(with_data))
        return with_data, nulls

    def calculate_similarities(self, with_data):
        i = 0
        index_sentence_map = {}
        for k in with_data:
            index_sentence_map[i] = k
            i += 1
        print("made index_sentence_map", len(index_sentence_map))
        mats = [m for m in self.mini_mats(with_data, 5000)]
        print("made mini_mats", len(mats))
        res = starmap(matrix_sim, combinations_with_replacement(mats, r=2))
        print("assigned batches with starmap")
        pc = 0
        pt = int((len(mats)*(len(mats)+1))/2)
        for r in res:
            pc += 1
            print("wrote batch {:>6,d}/{:>6,d}".format(pc, pt))

    def main(self):
        self.load_sentences()
        print("loaded sentences", len(self.sentences))

        with mp.Pool(mp.cpu_count() - 1) as pool:
            self.vectors = {k: v for k, v in (pool.map_async(func=self.process_sentence,
                                                             iterable=self.sentences.items())).get()}
            print("made sentence dict")

        with_data, nulls = self.filter_nulls()
        print("filtered nulls")

        self.calculate_similarities(with_data)
        print("finished similarities")

        fn = "/app/data/%s_LSA_%s.csv" % (TARGET, SPACE)
        with open(fn, "a") as o:
            for good, bad in product(self.sentences.keys(), nulls):
                line = "%s,%s,0.000\n" % (min(good, bad), max(good, bad))
                o.write(line)
        print("wrote nulls")
