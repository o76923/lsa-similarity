from datetime import datetime
from py.space_creator import Creator
from py.sim_calculator import LSASim
from py.configurator import ConfigSettings, CREATE_TASK, CALCULATE_TASK
from functools import partial


def echo_message(process, start, msg):
    print("{:<20}{:<20}{:<39}".format(process, str(datetime.now()-start), msg))

if __name__ == "__main__":
    start_time = datetime.now()
    cfg = ConfigSettings()
    echo_message("Delegator", start_time, "Loaded Configuration")
    for task in cfg.tasks:
        if task.type == CREATE_TASK:
            c = Creator(task, partial(echo_message, start=start_time))
            c.start = start_time
            c.main()
        if task.type == CALCULATE_TASK:
            s = LSASim(task, partial(echo_message, start=start_time))
            s.main()
        echo_message("Delegator", start_time, "Finished Task")
    echo_message("Delegator", start_time, "Done")

