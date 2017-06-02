import multiprocessing as mp
import os
import shutil

import h5py
import numpy as np

import py.vec_worker as vw
import py.document_cleaner as dc
from functools import partial
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from py.configurator import Project, SpaceSettings
from py.utils import *


class Projector(object):
    def __init__(self, config: Project, start_time):
        self._cfg = config
        self.dictionary = Dictionary.load("/app/data/spaces/%s/dictionary" % self._cfg.space_name)
        self.model = LsiModel.load("/app/data/spaces/%s/lsi" % self._cfg.space_name)
        self.raw_sentences = dict()
        self.sentences = dict()
        self.vectors = dict()
        self.announcer = partial(announcer, process="Projector", start=start_time)

    def load_sentences(self):
        self.raw_sentences = {}
        for source_file in self._cfg.source_files:
            with open("/app/data/{}".format(source_file)) as in_file:
                new_documents = {}
                if self._cfg.headers:
                    in_file.readline()

                if self._cfg.numbered:
                    for line in in_file.readlines():
                        sentence_id, sentence_text = line[:-1].split("\t")
                        new_documents[int(sentence_id)] = sentence_text
                else:
                    sentence_id = 0
                    for line in in_file:
                        new_documents[sentence_id] = line[:-1]
                        sentence_id += 1
                self.raw_sentences.update(new_documents)

    def load_space_settings(self):
        self._cfg.space_settings = SpaceSettings(space_name=self._cfg.space_name, load=True)

    def vectorize_sentences(self):
        with mp.Pool(self._cfg.num_cores, initializer=dc.init_worker, initargs=(self._cfg.space_settings,)) as pool:
            self.sentences = {k: v for k, v in pool.starmap_async(func=dc.clean_keyed_document,
                                                                  iterable=self.raw_sentences.items()).get()}
        with mp.Pool(self._cfg.num_cores, initializer=vw.init_worker, initargs=(self.dictionary, self.model)) as pool:
            self.vectors = {k: v for k, v in pool.starmap_async(func=vw.vectorize,
                                                                iterable=self.sentences.items()).get()}
        good_vectors = {k: v for k, v in self.vectors.items() if v.shape == (self._cfg.space_settings.dimensions, )}
        bad_vectors = {k: np.zeros(shape=(self._cfg.space_settings.dimensions, ), dtype=np.float32) for k, v in self.vectors.items() if v.shape != (self._cfg.space_settings.dimensions, )}
        self.vectors = good_vectors
        self.vectors.update(bad_vectors)
        # self.announcer("space_name: {}\n"
        #                "                                        output_file: {}\n"
        #                "                                        ds_name: {}\n"
        #                "                                        raw_sentences: {}\n"
        #                "                                        vectors: {}".format(self._cfg.space_name,
        #                                                                             self._cfg.output_file,
        #                                                                             self._cfg.ds_name,
        #                                                                             len(self.raw_sentences),
        #                                                                             len(self.vectors)))

    def save_hdf5(self):
        try:
            os.mkdir("/app/data/output")
            shutil.chown("/app/data/output/", user=1000)
        except FileExistsError:
            pass
        f = h5py.File("/app/data/output/{}".format(self._cfg.output_file), 'a')
        shutil.chown(f.filename, user=1000)
        sorted_vectors = [(k, v) for (k, v) in sorted(self.vectors.items(), key=lambda x: x[0])]
        # unused_keys = [k for k in self.raw_sentences if k not in [x[0] for x in sorted_vectors]]
        vec_array = np.stack([x[1] for x in sorted_vectors])
        string_dt = h5py.h5t.special_dtype(vlen=str)
        # float_dt = h5py.h5t.py_create(np.float16)
        vectors = f.require_group("vectors")
        vector = vectors.require_dataset(self._cfg.ds_name,
                                         dtype=np.float32,
                                         shape=vec_array.shape,
                                         data=vec_array,
                                         compression="gzip",
                                         compression_opts=9,
                                         shuffle=True,
                                         fillvalue=0.0
                                         )
        in_data = f.require_group("input")
        in_data.require_dataset("id",
                                dtype=np.uint16,
                                shape=(len(sorted_vectors),),
                                data=np.array([x[0] for x in sorted_vectors]))
        in_data.require_dataset("text",
                                dtype=string_dt,
                                shape=(len(sorted_vectors),),
                                data=[self.raw_sentences[x[0]].encode('utf-8') for x in sorted_vectors],
                                compression="gzip",
                                compression_opts=9,
                                shuffle=True)
        # if len(unused_keys) > 0:
        #     skipped = f.require_group("skip")
        #     skipped.require_dataset("id",
        #                             dtype=int,
        #                             shape=(len(unused_keys),),
        #                             data=unused_keys)
        f.flush()
        f.close()

    def main(self):
        self.load_space_settings()
        self.announcer("Loaded Space Settings")
        self.load_sentences()
        self.announcer("Loaded Sentences")
        self.vectorize_sentences()
        self.announcer("Vectorized Sentences")
        self.save_hdf5()
        self.announcer("Saved into HDF5 format")
