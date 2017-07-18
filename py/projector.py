import multiprocessing as mp
import os
import shutil
from functools import partial

import h5py
from gensim.corpora import Dictionary
from gensim.models import LsiModel

import py.document_cleaner as dc
import py.vec_worker as vw
from py.configurator import Project, SpaceSettings
from py.utils import *


class Projector(object):
    def __init__(self, config: Project, start_time):
        self._cfg = config
        self.dictionary = Dictionary.load("/app/data/spaces/{}/dictionary".format(self._cfg.space_name))
        self.model = LsiModel.load("/app/data/spaces/{}/lsi".format(self._cfg.space_name))
        try:
            self.rot_mat = np.load("/app/data/spaces/{}/rot_mat.npy".format(self._cfg.space_name))
        except FileNotFoundError:
            pass
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
        if self._cfg.rotated:
            with mp.Pool(self._cfg.num_cores, initializer=vw.init_worker,
                         initargs=(self.dictionary, self.model, self.rot_mat)) as pool:
                self.vectors = {k: v for k, v in pool.starmap_async(func=vw.rotated_vectorize,
                                                                    iterable=self.sentences.items()).get()}
        else:
            with mp.Pool(self._cfg.num_cores, initializer=vw.init_worker,
                         initargs=(self.dictionary, self.model)) as pool:
                self.vectors = {k: v for k, v in pool.starmap_async(func=vw.vectorize,
                                                                    iterable=self.sentences.items()).get()}
        good_vectors = {k: v for k, v in self.vectors.items() if v.shape == (self._cfg.space_settings.dimensions, )}
        bad_vectors = {k: np.zeros(shape=(self._cfg.space_settings.dimensions, ), dtype=np.float32) for k, v in self.vectors.items() if v.shape != (self._cfg.space_settings.dimensions, )}
        self.vectors = good_vectors
        self.vectors.update(bad_vectors)

    def save_hdf5(self):
        try:
            os.mkdir("/app/data/output")
            shutil.chown("/app/data/output/", user=1000)
        except FileExistsError:
            pass
        if self._cfg.output_format == OUTPUT_FORMAT.H5:
            f = h5py.File('/app/data/output/{}'.format(self._cfg.output_file), 'a')
        else:
            f = h5py.File('{}/{}.h5'.format(self._cfg.temp_dir, self._cfg.output_file), 'a')
        shutil.chown(f.filename, user=1000)
        sorted_vectors = [(k, v) for (k, v) in sorted(self.vectors.items(), key=lambda x: x[0])]
        # unused_keys = [k for k in self.raw_sentences if k not in [x[0] for x in sorted_vectors]]
        vec_array = np.stack([x[1] for x in sorted_vectors])
        string_dt = h5py.h5t.special_dtype(vlen=str)
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
                                dtype='u8',
                                shape=(len(sorted_vectors),),
                                data=np.array([x[0] for x in sorted_vectors]))
        in_data.require_dataset("text",
                                dtype=string_dt,
                                shape=(len(sorted_vectors),),
                                data=[self.raw_sentences[x[0]].encode('utf-8') for x in sorted_vectors],
                                compression="gzip",
                                compression_opts=9,
                                shuffle=True)
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
