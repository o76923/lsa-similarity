from functools import partial
from py.configurator import Rotate
from py.utils import *


class Rotator(object):

    def __init__(self, config: Rotate, start_time):
        self._cfg = config
        self.announcer = partial(announcer, process="Converter", start=start_time)

    def rotate(self):
        pass

    def main(self):
        self.rotate()
