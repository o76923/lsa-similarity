import multiprocessing as mp
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LogEntropyModel, LsiModel
import re


class Creator(object):

    def __init__(self, paragraph_file, space_name):
        self.stemmer = PorterStemmer()
        self.rank = 300
        self.alnum_patt = re.compile('[^a-z0-9]')
        self.stopwords = set(stopwords.words('english'))
        self.paragraph_file = paragraph_file
        self.space_name = space_name
        self.documents = list()

    def load_documents(self):
        with open("/app/data/%s.txt" % self.paragraph_file) as in_file:
            in_file.readline()
            self.documents = in_file.readlines()

    def _try_stem(self, word):
        try:
            return self.stemmer.stem(word)
        except IndexError:
            return ""

    def clean_documents(self, sentence):
        try:
            return [w for w in [self._try_stem(w) for w in self.alnum_patt.sub(' ', sentence.lower()).split() if w not in self.stopwords] if len(w) > 0]
        except AttributeError:
            return []

    def create_dictionary(self, clean_documents):
        dictionary = Dictionary()
        dictionary.add_documents(clean_documents, prune_at=None)
        print("Made Dictionary")
        singleton_oids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(singleton_oids)
        dictionary.compactify()
        print("Filtered Dictionary")
        return dictionary

    def create_space(self, clean_documents):
        dictionary = self.create_dictionary(clean_documents)
        corpus = [dictionary.doc2bow(paragraph) for paragraph in clean_documents]
        print("Created Corpus")
        log_ent_model = LogEntropyModel(corpus, id2word=dictionary)
        print("Made Log Entropy Model")
        log_ent_corpus = log_ent_model[corpus]
        print("Made Log Entropy Corpus")
        lsa_model = LsiModel(log_ent_corpus, id2word=dictionary, num_topics=self.rank, distributed=False)
        print("Made LSA Model")
        return dictionary, lsa_model

    def main(self):
        self.load_documents()
        print("loaded documents")
        pool = mp.Pool(mp.cpu_count()-1)
        clean_documents = [l for l in pool.map_async(func=self.clean_documents, iterable=self.documents).get()]
        pool.close()
        print("Cleaned documents")
        dictionary, model = self.create_space(clean_documents)
        print("Created space")
        dictionary.save('/app/data/%s.dictionary' % self.space_name)
        print("Saved Dictionary")
        model.save('/app/data/%s.lsi' % self.space_name)
        print("Saved LSA Model")
