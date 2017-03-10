import yaml
from py.space_settings import SpaceSettings


class ConfigSettings(object):
    space_name: str
    space_files: list
    sentence_files: list
    pair_mode = str
    pair_list: list
    num_cores: int
    max_memory: int

    def __init__(self, filename="/app/data/config.yml"):
        self._read_config(filename)

        self.create_space = "create_space" in self._cfg['tasks']
        if self.create_space:
            self.space_name = self._cfg['tasks']['create_space']['name']
            self._initialize_space_settings('create_space')
            self._initialize_space_files()

        self.calculate_sims = "calculate_sims" in self._cfg['tasks']
        if self.calculate_sims:
            self.space_name = self._cfg['tasks']['calculate_sims']['space']
            self._read_space_settings("/app/data/spaces/%s/space_config.yml" % self.space_name)
            self._initialize_space_settings('calculate_sims')
            self._initialize_sentence_files()
            self._initialize_pair_mode()
            self._initialize_calculate_settings()

        self._initialize_global_options()

    def _read_config(self, filename):
        with open(filename) as in_file:
            self._cfg = yaml.load(in_file.read())

    def _read_space_settings(self, filename):
        if 'create_space' in self._cfg['tasks']:
            self._cfg['tasks']['calculate_sims']['space_settings'] = self._cfg['tasks']['create_space']['space_settings']
        else:
            with open(filename) as in_file:
                self._cfg['tasks']['calculate_sims']['space_settings'] = yaml.load(in_file.read())

    def _initialize_space_files(self):
        self.space_files = []
        if "files" in self._cfg['tasks']['create_space']['from']:
            self.space_files = self._cfg['tasks']['create_space']['from']['files']

    def _initialize_sentence_files(self):
        self.sentence_files = []
        if "files" in self._cfg['tasks']['calculate_sims']['from']:
            # self.sentence_files = ["".join(x) for x in self._cfg['tasks']['calculate_sims']['from']['files']]
            self.sentence_files = self._cfg['tasks']['calculate_sims']['from']['files']

    def _initialize_space_settings(self, source):

        self.space_settings = SpaceSettings()

        # check things with default true
        # make them true if absent, present without a value, or present with true and false if present with false
        for param in ("stem", "case_insensitive"):
            if param in self._cfg['tasks'][source]['space_settings']:
                try:
                    self.space_settings[param] = self._cfg['tasks'][source]['space_settings'][param]
                except AttributeError:
                    self.space_settings[param] = True
            else:
                self.space_settings[param] = True
        
        # check things that should be removed
        # set to remove if present or not if absent
        for param in ("punctuation", "numbers", "singletons"):
            if param in self._cfg['tasks'][source]['space_settings']['remove']:
                self.space_settings['remove_%s' % param] = True
            else:
                self.space_settings['remove_%s' % param] = False
        
        # pull out a list of stopwords from the nltk library
        for r in self._cfg['tasks'][source]['space_settings']['remove']:
            try:
                if 'stopwords' in r.keys():
                    from nltk.corpus import stopwords
                    self.space_settings.stopwords = stopwords.words('english')
            except AttributeError:
                pass

        # set dimensions, default 300
        if 'dimensions' in self._cfg['tasks'][source]['space_settings']:
            self.space_settings.dimensions = self._cfg['tasks'][source]['space_settings']['dimensions']
        else:
            self.space_settings.dimensions = 300

    def _initialize_pair_mode(self):
        if 'pairs' in self._cfg['tasks']['calculate_sims']['from']:
            pair_mode = self._cfg['tasks']['calculate_sims']['from']['pairs']
            if pair_mode in ('all', 'cross'):
                self.pair_mode = pair_mode
            else:
                self.pair_mode = 'list'
                self.pair_list = []
                for p in self._cfg['tasks']['calculate_sims']['from']['pairs']:
                    with open("/app/data/%s" % p) as in_file:
                        new_pairs = [x for x in in_file.readlines()]
                        self.pair_list.extend(new_pairs)
        else:
            self.pair_mode = 'all'

    def _initialize_calculate_settings(self):
        try:
            self.sim_batch_size = self._cfg['tasks']['calculate_sims']['options']['batch_size']
        except KeyError:
            self.sim_batch_size = 100

    def _initialize_global_options(self):
        try:
            self.num_cores = self._cfg['options']['cores']
        except KeyError:
            from multiprocessing import cpu_count
            self.num_cores = cpu_count()