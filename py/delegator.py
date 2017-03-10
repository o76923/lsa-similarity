from datetime import datetime
from py.space_creator import Creator
from py.sim_calculator import LSASim
from py.configurator import ConfigSettings
from functools import partial


def echo_message(process, start, msg):
    print("{:<20}{:<20}{:<39}".format(process, str(datetime.now()-start), msg))

if __name__ == "__main__":
    start = datetime.now()
    cfg = ConfigSettings()
    echo_message("Delegator", start, "Loaded Configuration")
    if cfg.create_space:
        c = Creator(cfg, partial(echo_message, start=start))
        c.start = start
        c.main()

    if cfg.calculate_sims:
        s = LSASim(cfg, partial(echo_message, start=start))
        s.main()
    echo_message("Delegator", start, "Done")
