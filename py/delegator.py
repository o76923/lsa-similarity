import os

from py.space_creator import Creator
from py.projector import Projector
from py.sim_calculator import SimCalculator
from py.format_converter import FormatConverter
from py.configurator import Config
from py.utils import *
from functools import partial


if __name__ == "__main__":
    start_time = datetime.now()
    announcer = partial(announcer, process="Delegator", start=start_time)
    cfg = Config()
    announcer("Loaded Configuration")

    os.makedirs(cfg.temp_dir)
    announcer("Created temp directory at {}".format(cfg.temp_dir))

    for task in cfg.tasks:
        if task.type == TASK_TYPE.CREATE:
            t = Creator(task, start_time)
        elif task.type == TASK_TYPE.PROJECT:
            t = Projector(task, start_time)
        elif task.type == TASK_TYPE.CALCULATE:
            t = SimCalculator(task, start_time)
        elif task.type == TASK_TYPE.CONVERT:
            t = FormatConverter(task, start_time)
        else:
            raise Exception("Illegal task_type")
        t.main()
        announcer("Finished Task")
    announcer("Done")

