import yaml

CREATE_TASK = 0
CALCULATE_TASK = 1


class SpaceSettings(object):
    dimensions: int
    remove_punctuation: bool
    remove_numbers: bool
    remove_singletons: bool
    stem: bool
    case_insensitive: bool
    stopwords: list

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            return False

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def save(self, path):
        with open("%s/space_config.yml" % path, "w") as out_file:
            data = {
                "dimensions": self.dimensions,
                "stem": self.stem,
                "case_insensitive": self.case_insensitive,
                "remove": []
            }
            if self.remove_punctuation:
                data["remove"].append("punctuation")
            if self.remove_numbers:
                data["remove"].append("numbers")
            if self.remove_singletons:
                data["remove"].append("singletons")
            if self.stopwords:
                data['stopwords'] = True
                with open("%s/stopwords.txt" % path, "w") as sw_file:
                    sw_file.write("\n".join(self.stopwords))
            else:
                data['stopwords'] = False
            yaml.dump(data, out_file)


class TaskSettings(object):
    space_name: str
    space_settings: SpaceSettings
    type: str
    num_cores: int


class CreateSettings(TaskSettings):
    space_files: list
    type = CREATE_TASK


class CalculateSettings(TaskSettings):
    sentence_files: list
    pair_mode: str
    pair_list: list
    file_name: str
    top_n: int
    input_headers: bool
    sim_batch_size: int
    output_null: str
    type = CALCULATE_TASK


class ConfigSettings(object):

    def __init__(self, filename="/app/data/config.yml"):
        self._read_config(filename)
        self.tasks = []

        for t in self._cfg['tasks']:
            if t["type"] == "create_space":
                task = CreateSettings()
                task.space_name = t['name']
                task.space_settings = self._initialize_space_settings(t)
                task.space_files = self._initialize_space_files(t)
            elif t["type"] == "calculate_sims":
                task = CalculateSettings()
                task.space_name = t['space']
                try:
                    task.space_settings = self.tasks[0].space_settings
                except IndexError:
                    t['space_settings'] = self._read_space_settings(task.space_name)
                    task.space_settings = self._initialize_space_settings(t)
                task.sentence_files = self._initialize_sentence_files(t)
                task.input_headers = self._initialize_input_headers(t)
                task.pair_mode, task.pair_list = self._initialize_pair_mode(t)
                task.sim_batch_size = self._initialize_calculate_settings(t)
                task.file_name, task.top_n = self._initialize_calculate_output(t)
                task.output_null = self._initialize_null_output(t)
            task.num_cores = self._initialize_global_options()
            self.tasks.append(task)

    def _read_config(self, filename):
        with open(filename) as in_file:
            self._cfg = yaml.load(in_file.read())

    def _read_space_settings(self, space_name):
        with open("/app/data/spaces/%s/space_config.yml" % space_name) as in_file:
            space_settings = yaml.load(in_file.read())
        return space_settings

    @staticmethod
    def _initialize_space_files(t):
        space_files = []
        if "files" in t['from']:
            space_files = t['from']['files']
        return space_files

    @staticmethod
    def _initialize_sentence_files(t):
        sentence_files = []
        if "files" in t['from']:
            sentence_files = t['from']['files']
        return sentence_files

    @staticmethod
    def _initialize_input_headers(t):
        try:
            if "headers" in t['from']:
                return t['from']['headers']
        except KeyError:
            pass
        return False

    @staticmethod
    def _initialize_null_output(t):
        try:
            if "nulls" in t['output']:
                return str(t['output']['nulls'])
        except KeyError:
            pass
        return "NULL"

    @staticmethod
    def _initialize_space_settings(t):

        space_settings = SpaceSettings()

        # check things with default true
        # make them true if absent, present without a value, or present with true and false if present with false
        for param in ("stem", "case_insensitive"):
            if param in t['space_settings']:
                try:
                    space_settings[param] = t['space_settings'][param]
                except AttributeError:
                    space_settings[param] = True
            else:
                space_settings[param] = True
        
        # check things that should be removed
        # set to remove if present or not if absent
        for param in ("punctuation", "numbers", "singletons"):
            if param in t['space_settings']['remove']:
                space_settings['remove_%s' % param] = True
            else:
                space_settings['remove_%s' % param] = False
        
        # pull out a list of stopwords from the nltk library
        for r in t['space_settings']['remove']:
            try:
                if 'stopwords' in r.keys():
                    from nltk.corpus import stopwords
                    space_settings.stopwords = stopwords.words('english')
            except AttributeError:
                pass

        # set dimensions, default 300
        if 'dimensions' in t['space_settings']:
            space_settings.dimensions = t['space_settings']['dimensions']
        else:
            space_settings.dimensions = 300

        return space_settings

    @staticmethod
    def _initialize_pair_mode(t):
        pair_list = []
        if 'pairs' in t['from']:
            pair_mode = t['from']['pairs']
            if pair_mode in ('all', 'cross'):
                pair_mode = pair_mode
            else:
                pair_mode = 'list'
                for p in t['from']['pairs']:
                    with open("/app/data/%s" % p) as in_file:
                        new_pairs = [x for x in in_file.readlines()]
                        pair_list.extend(new_pairs)
        else:
            pair_mode = 'all'
        return pair_mode, pair_list

    @staticmethod
    def _initialize_calculate_settings(t):
        try:
            sim_batch_size = t['options']['batch_size']
        except KeyError:
            sim_batch_size = 100
        return sim_batch_size

    @staticmethod
    def _initialize_calculate_output(t):
        try:
            sim_file_name = t['output']['filename']
        except KeyError:
            sim_file_name = "sims.csv"
        top_n = None
        try:
            left_min_sim = t['output']['similarity_count']['left']
            top_n = int(left_min_sim)
        except KeyError:
            try:
                min_sim = int(t['output']['similarity_count'])
                top_n = min_sim
            except (TypeError, KeyError):
                pass
        return sim_file_name, top_n

    def _initialize_global_options(self):
        try:
            return self._cfg['options']['cores']
        except KeyError:
            from multiprocessing import cpu_count
            return cpu_count()
