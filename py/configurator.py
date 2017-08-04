import multiprocessing as mp
import os
import warnings
from typing import List, Optional, Text

import yaml

from py.utils import *

CONFIG_FILE = "/app/data/"+os.environ.get("CONFIG_FILE", "config.yml")


class SpaceSettings(object):
    space_name: Text
    dimensions: int
    stem: bool
    case_sensitive: bool
    remove_punctuation: bool
    remove_numbers: bool
    remove_singletons: bool
    remove_stopwords: bool
    stopwords: Optional[Text]

    def __init__(self, space_name, load=False, **kwargs):
        self.space_name = space_name
        if load:
            with open("/app/data/spaces/{}/space_config.yml".format(space_name)) as in_file:
                kwargs = yaml.load(in_file)
        try:
            self.dimensions = kwargs["dimensions"]
        except:
            raise Exception("Number of dimensions not specified")

        try:
            self.stem = kwargs["stem"]
        except KeyError:
            self.stem = False

        try:
            self.case_sensitive = kwargs["case_sensitive"]
        except KeyError:
            self.case_sensitive = False

        try:
            if "punctuation" in kwargs["remove"]:
                self.remove_punctuation = True
            else:
                self.remove_punctuation = False
            if "numbers" in kwargs["remove"]:
                self.remove_numbers = True
            else:
                self.remove_numbers = False
            if "singletons" in kwargs["remove"]:
                self.remove_singletons = True
            else:
                self.remove_singletons = False
            if "stopwords" in kwargs["remove"]:
                self.remove_stopwords = True
                with open("{}".format(kwargs["remove"]["stopwords"])) as stopwords_file:
                    self.stopwords = [line[:-1] for line in stopwords_file]

            else:
                self.remove_stopwords = False
                self.stopwords = None
        except KeyError:
            pass

    def save(self):
        with open("/app/data/spaces/{}/space_config.yml".format(self.space_name), "w") as out_file:
            data = {
                "dimensions": self.dimensions,
                "stem": self.stem,
                "case_sensitive": self.case_sensitive,
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
                with open("%s/stopwords.txt" % self.space_name, "w") as sw_file:
                    sw_file.write("\n".join(self.stopwords))
            else:
                data['stopwords'] = False
            yaml.dump(data, out_file)


class Task(object):
    num_cores: int
    # temp_dir: Text
    type: TASK_TYPE

    def __init__(self, global_settings):
        self.num_cores = global_settings["num_cores"]
        # self.temp_dir = global_settings["temp_dir"]


class Create(Task):
    space_name: Text
    source_files: List[Text]
    space_settings: SpaceSettings
    headers: bool
    numbered: bool
    compute_rotation: bool

    def __init__(self, global_settings, task_settings):
        super().__init__(global_settings)
        self.type = TASK_TYPE.CREATE
        self.source_files = task_settings["from"]["files"]
        self.space_name = task_settings["options"]["space"]
        try:
            self.headers = task_settings["from"]["headers"]
        except KeyError:
            self.headers = False
        try:
            self.numbered = task_settings["from"]["numbered"]
        except KeyError:
            self.numbered = False
        self.space_settings = SpaceSettings(space_name=task_settings["options"]["space"],
                                            load=False,
                                            dimensions=task_settings["options"]["dimensions"],
                                            stem=task_settings["options"]["stem"],
                                            remove=task_settings["options"]["remove"],
                                            case_sensitive=task_settings["options"]["case_sensitive"])
        try:
            self.compute_rotation = task_settings["options"]["compute_varimax"]
        except KeyError:
            self.compute_rotation = False
            warnings.warn("No compute_varimax option specified, not computing")


class Rotate(Task):
    space_name: Text
    num_dims: int
    kaiser_norm: bool

    def __init__(self, global_settings, task_settings):
        super().__init__(global_settings)
        self.type = TASK_TYPE.ROTATE
        try:
            self.space_name = task_settings["options"]["space"]
        except KeyError:
            raise Exception("You must specify which space that you want to rotate")
        try:
            self.num_dims = task_settings["options"]["dimensions"]
        except KeyError:
            warnings.warn("Number of dimensions not specified. Using all dimensions in the space.")
        try:
            self.kaiser_norm = task_settings["options"]["normalize"]
        except KeyError:
            self.kaiser_norm = True
            warnings.warn("Whether to normalize the rotation is unspecified, defaulting to true")


class Project(Task):
    space_name: Text
    source_files: List[Text]
    source_names: List[Text]
    space_settings: Optional[SpaceSettings]
    headers: bool
    numbered: bool
    rotated: bool
    num_dims: int
    output_format: OUTPUT_FORMAT
    output_file: Optional[Text]

    def __init__(self, global_settings, task_settings):
        super().__init__(global_settings)
        self.type = TASK_TYPE.PROJECT
        self.source_files = []
        self.source_names = []
        for file in task_settings["from"]["files"]:
            try:
                self.source_files.append(file["file_name"])
                self.source_names.append(file["source_name"])
            except (TypeError, KeyError):
                self.source_files.append(file)
        try:
            self.space_name = task_settings["options"]["space"]
        except KeyError:
            raise Exception("A semantic space must be specified.")
        try:
            self.headers = task_settings["from"]["headers"]
        except KeyError:
            self.headers = False
        try:
            self.numbered = task_settings["from"]["numbered"]
        except KeyError:
            self.numbered = False
        try:
            self.rotated = task_settings["options"]["rotated"]
        except KeyError:
            self.rotated = False
            warnings.warn("Rotation not specified, defaulting to unrotated projection")
        try:
            self.num_dims = task_settings["options"]["dimensions"]
        except KeyError:
            warnings.warn("Number of dimensions not specified, defaulting to all in the space.")
        if "output" in task_settings:
            try:
                self.output_format = OUTPUT_FORMAT[task_settings["output"]["format"].upper()]
            except KeyError:
                self.output_format = OUTPUT_FORMAT.H5
            try:
                self.output_file = task_settings["output"]["file_name"]
            except:
                raise Exception("You must specify an output file_name when saving output")
            try:
                self.ds_name = task_settings["output"]["ds_name"]
            except KeyError:
                self.ds_name = 'sim'
                warnings.warn("No ds_name specified, using 'sim' as name of data source in vectors")


class Calculate(Task):
    space_name: Text
    distance_metric: DISTANCE_METRIC
    output_file: Text
    ds_name: Text
    output_format: OUTPUT_FORMAT
    pair_mode: PAIR_MODE

    def __init__(self, global_settings, task_settings):
        super().__init__(global_settings)
        self.type = TASK_TYPE.CALCULATE

        try:
            self.pair_mode = PAIR_MODE[task_settings["from"]["pairs"].upper()]
        except KeyError:
            warnings.warn("No pair mode specified, using 'all' by default.")
            self.pair_mode = PAIR_MODE.ALL

        if self.pair_mode == PAIR_MODE.ALL:
            global_settings["tasks"].append(Project(global_settings, task_settings))
        elif self.pair_mode == PAIR_MODE.CROSS:
            for f in task_settings["from"]["files"]:
                subtask_settings = task_settings.copy()
                subtask_settings["from"]["files"] = f
                global_settings["tasks"].append(Project(global_settings, subtask_settings))
        else:
            raise Exception("You have specified an illegal pair mode, please use 'all' or 'cross'")

        try:
            self.space_name = task_settings["options"]["space"]
        except KeyError:
            raise Exception("A semantic space must be specified.")
        try:
            self.distance_metric = DISTANCE_METRIC[task_settings["options"]["distance_metric"].upper()]
            if self.distance_metric == DISTANCE_METRIC.R:
                raise Exception("Correlation distance metric not yet implemented")
        except KeyError:
            warnings.warn("Illegal distance metric specified. Using cosine similarity instead.")
        try:
            self.output_format = OUTPUT_FORMAT[task_settings["output"]["format"]]
        except KeyError:
            self.output_format = OUTPUT_FORMAT.H5
        try:
            self.output_file = task_settings["output"]["file_name"]
        except KeyError:
            raise Exception("You must specify an output file_name when saving output")
        try:
            self.ds_name = task_settings["output"]["ds_name"]
        except KeyError:
            self.ds_name = 'sim'
            warnings.warn("No ds_name specified, using 'sim' as name of data source in sims")


class Config(object):
    tasks: List[Task]
    # temp_dir: str
    num_cores: int

    def __init__(self):
        self._read_config(CONFIG_FILE)
        self._load_global()
        # self.temp_dir = "/tmp/lsa_{}".format(uuid4())
        self.tasks = []

        global_settings = {
            # "temp_dir": self.temp_dir,
            "num_cores": self.num_cores,
            "tasks": self.tasks
        }

        for task in self._cfg['tasks']:
            self.tasks.append(self._load_task(global_settings, task))

    def _read_config(self, filename):
        with open(filename) as in_file:
            self._cfg = yaml.load(in_file.read())

    def _load_global(self):
        try:
            self.num_cores = int(self._cfg["options"]["cores"])
            return
        except KeyError:
            warnings.warn("Number of cores not specified, defaulting to one less than max")
        except TypeError:
            warnings.warn("The number of cores must be an int, defaulting to one less than max insetad")
        finally:
            self.num_cores = max(mp.cpu_count() - 1, 1)

    def _load_task(self, global_settings, task_settings):
        try:
            if task_settings["type"] == "create_space":
                return Create(global_settings, task_settings)
            elif task_settings["type"] == "project_sentences":
                return Project(global_settings, task_settings)
            elif task_settings["type"] == "rotate_space":
                return Rotate(global_settings, task_settings)
            elif task_settings["type"] == "calculate_similarity":
                return Calculate(global_settings, task_settings)
            else:
                raise Exception("Invalid task type supplied")
        except KeyError:
            raise Exception("No task type specified")