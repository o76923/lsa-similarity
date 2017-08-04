import multiprocessing as mp
import os
from functools import partial

from gensim.corpora import Dictionary
from gensim.models import LogEntropyModel, LsiModel

from py.configurator import Create
from py.document_cleaner import clean_document, init_worker
from py.utils import *


class Creator(object):

    def __init__(self, config: Create, start_time):
        self._cfg = config
        self.documents = list()
        self.announcer = partial(announcer, process="Creator", start=start_time)

    def load_documents(self):
        self.documents = []
        for paragraph_file in self._cfg.source_files:
            with open("/app/data/%s" % paragraph_file) as in_file:
                if self._cfg.headers:
                    in_file.readline()
                if self._cfg.numbered:
                    new_documents = [l[:-1].split("\t")[1] for l in in_file.readlines()]
                else:
                    new_documents = in_file.readlines()
                self.documents.extend(new_documents)

    def create_dictionary(self, clean_documents):
        dictionary = Dictionary()
        dictionary.add_documents(clean_documents, prune_at=None)
        self.announcer("Loaded Dictionary")
        if self._cfg.space_settings.remove_singletons:
            singleton_oids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
            dictionary.filter_tokens(singleton_oids)
        dictionary.compactify()
        self.announcer("Filtered Dictionary")
        return dictionary

    def create_space(self, clean_documents):
        dictionary = self.create_dictionary(clean_documents)
        corpus = [dictionary.doc2bow(paragraph) for paragraph in clean_documents]
        self.announcer("Created Corpus")
        log_ent_model = LogEntropyModel(corpus, id2word=dictionary)
        self.announcer("Made Log Entropy Model")
        log_ent_corpus = log_ent_model[corpus]
        self.announcer("Made Log Entropy Corpus")
        lsa_model = LsiModel(log_ent_corpus,
                             id2word=dictionary,
                             num_topics=self._cfg.space_settings.dimensions,
                             distributed=False)
        self.announcer("Made LSA Model")
        return dictionary, lsa_model

    def save_all(self, dictionary, model):
        space_dir = '/app/data/spaces/{}'.format(self._cfg.space_name)
        try:
            os.makedirs(space_dir)
        except FileExistsError:
            pass
        dictionary.save('{}/dictionary'.format(space_dir))
        self.announcer("Saved Dictionary")
        model.save('{}/lsi'.format(space_dir))
        self.announcer("Saved LSA Model")
        self._cfg.space_settings.save()
        self.announcer("Saved Settings")

    def main(self):
        self.announcer("Started")
        self.load_documents()
        self.announcer("Loaded Documents")
        with mp.Pool(mp.cpu_count() - 1, initializer=init_worker, initargs=(self._cfg.space_settings,)) as pool:
            clean_documents = [l for l in pool.map_async(func=clean_document, iterable=self.documents).get()]
        self.announcer("Cleaned Documents")
        dictionary, model = self.create_space(clean_documents)
        self.announcer("Created Space")
        self.save_all(dictionary, model)
        self.announcer("Finished saving")
