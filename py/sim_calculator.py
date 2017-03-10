import string
from itertools import product, combinations_with_replacement, count
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import numpy as np
import re
import multiprocessing as mp
from py.configurator import ConfigSettings
import py.document_cleaner as dc
import py.vectorizer as vw
import py.sim_batch_worker as sbw
import subprocess
from functools import partial
from shutil import rmtree
from os import mkdir
from collections import OrderedDict


class LSASim(object):
    PUNCTUATION_PATTERN = re.compile('[%s]' % re.escape(string.punctuation))

    def __init__(self, config: ConfigSettings, announcer):
        self._cfg = config
        self.dictionary = Dictionary.load("/app/data/spaces/%s/dictionary" % self._cfg.space_name)
        self.model = LsiModel.load("/app/data/spaces/%s/lsi" % self._cfg.space_name)
        self._load_stopwords("/app/data/spaces/%s/stopwords.txt" % self._cfg.space_name)
        self.sentences = dict()
        self.vectors = dict()
        self.nulls = set()
        self.file_keys = OrderedDict()
        self.file_nulls = OrderedDict()
        self.announcer = partial(announcer, process="Calculator")

    def _load_stopwords(self, filename):
        with open(filename) as in_file:
            self.stopwords = [l[:-1] for l in in_file.readlines()]

    def create_temp_dir(self):
        try:
            rmtree("/app/data/temp_sim")
        except FileNotFoundError:
            pass
        mkdir("/app/data/temp_sim")

    def load_sentences(self):
        sentences = {}
        for sentence_file in self._cfg.sentence_files:
            with open("/app/data/%s" % sentence_file) as in_file:
                new_documents = {}
                for line in in_file.readlines():
                    try:
                        data = line[:-1].split("\t")
                        new_documents[data[0]] = data[1]
                    except IndexError:
                        print(line)
                sentences.update(new_documents)
                if self._cfg.pair_mode == 'cross':
                    self.file_keys[sentence_file] = sorted(list(new_documents.keys()))
        with mp.Pool(mp.cpu_count()-1, initializer=dc.init_worker, initargs=(self._cfg.space_settings,)) as pool:
            self.sentences = {k: v for k, v in pool.starmap_async(func=dc.clean_keyed_document, iterable=sentences.items()).get()}

    def vectorize_sentences(self):
        with mp.Pool(mp.cpu_count()-1, initializer=vw.init_worker, initargs=(self.dictionary, self.model)) as pool:
            self.vectors = {k: v for k, v in pool.starmap_async(func=vw.vectorize, iterable=self.sentences.items()).get()}

    def filter_nulls(self):
        self.nulls = set(k for k in self.vectors if self.vectors[k] is None or len(self.vectors[k]) != self._cfg.space_settings.dimensions)
        self.vectors = {k: self.vectors[k] for k in (set(self.vectors.keys()) - self.nulls)}
        try:
            self.file_nulls = OrderedDict([(k, sorted(list(set(v).intersection(self.nulls)))) for k, v in self.file_keys.items()])
            self.file_keys = OrderedDict([(k, sorted(list(set(v)-self.nulls))) for k, v in self.file_keys.items()])
        except NameError:
            pass

    def _make_small_mats(self, file_name=None):
        if not file_name:
            vector_keys = sorted(list(self.vectors.keys()))
        else:
            vector_keys = self.file_keys[file_name]
        for i in range(0, len(vector_keys), self._cfg.sim_batch_size):
            keys = [k for k in vector_keys[i:i+self._cfg.sim_batch_size]]
            mat = np.array([self.vectors[k] for k in keys])
            yield keys, mat

    def calculate_similarities(self):
        if self._cfg.pair_mode == 'all':
            small_mat = [m for m in self._make_small_mats()]
            batches = combinations_with_replacement(small_mat, r=2)
            batch_count = (len(self.vectors)//self._cfg.sim_batch_size+1)*(len(self.vectors)//self._cfg.sim_batch_size)//2+1
            self.announcer(msg="Made %d small mats for %d batches" % (len(small_mat), batch_count))
        elif self._cfg.pair_mode == 'cross':
            left_file, right_file = self.file_keys.keys()
            left_batches = [l for l in self._make_small_mats(left_file)]
            right_batches = [r for r in self._make_small_mats(right_file)]
            batches = product(left_batches, right_batches)
            batch_count = len(left_batches) * len(right_batches)
            self.announcer(msg="Made %d left mats and %d right mats for %d batches" % (len(left_batches), len(right_batches), batch_count))
        else:
            batches = []
            batch_count = 0
        with mp.Pool(14, initializer=sbw.init_worker, initargs=(self.announcer, batch_count)) as pool:
            pool.starmap_async(sbw.process_batch, [(b[0], b[1], c) for b, c in zip(batches, count(start=1))], chunksize=10).get()

    def merge_similarity_files(self):
        subprocess.run("cat /app/data/temp_sim/sims-*.csv > /app/data/sims.csv", shell=True)
        self.announcer(msg="Catted all files to sims.csv")
        rmtree("/app/data/temp_sim")
        self.announcer(msg="Removed temp_sims")

    def append_null_sims(self):
        lines = []
        if self._cfg.pair_mode == 'all':
            pass
        elif self._cfg.pair_mode == 'cross':
            left_nulls, right_nulls = self.file_nulls.values()
            for l, r in product(left_nulls, right_nulls):
                    lines.append("{},{},0.0\n".format(l, r))
        with open("/app/data/sims.csv", "a") as out_file:
            out_file.write("".join(lines))

    def main(self):
        self.announcer(msg="Started")
        self.create_temp_dir()
        self.announcer(msg="Created empty temp directory")
        self.load_sentences()
        self.announcer(msg="Loaded %d sentences" % len(self.sentences))
        self.vectorize_sentences()
        self.announcer(msg="Vectorized sentences")
        self.filter_nulls()
        self.announcer(msg="Filtered nulls; %d removed, %d retained" % (len(self.nulls), len(self.vectors)))
        self.calculate_similarities()
        self.announcer(msg="Calculated all similarities")
        self.merge_similarity_files()
        self.announcer(msg="Merged files")
        self.append_null_sims()
        self.announcer(msg="Appended Nulls")