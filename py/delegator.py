from functools import partial

from py.configurator import Config
from py.projector import Projector
from py.rotator import Rotator
from py.sim_calculator import SimCalculator
from py.space_creator import Creator
from py.utils import *


start_time = datetime.now()
announcer = partial(announcer, process="Delegator", start=start_time)
cfg = Config()
announcer("Loaded Configuration")

# os.makedirs(cfg.temp_dir)
# announcer("Created temp directory at {}".format(cfg.temp_dir))

for task in cfg.tasks:
    if task.type == TASK_TYPE.CREATE:
        t = Creator(task, start_time)
    elif task.type == TASK_TYPE.ROTATE:
        t = Rotator(task, start_time)
    elif task.type == TASK_TYPE.PROJECT:
        t = Projector(task, start_time)
    elif task.type == TASK_TYPE.CALCULATE:
        t = SimCalculator(task, start_time)
    else:
        raise Exception("Illegal task_type")
    t.main()
    announcer("Finished Task")
announcer("Done")
