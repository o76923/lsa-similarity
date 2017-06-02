from functools import partial
from py.configurator import Convert
from py.utils import *
import h5py as h5


class FormatConverter(object):

    def __init__(self, config: Convert, start_time):
        self._cfg = config
        self.announcer = partial(announcer, process="Converter", start=start_time)

    def convert(self):
        f = h5.File('/app/data/output/{}'.format(self._cfg.output_file))
        ids = f.get('/input/id')
        node = f.get('/sim/{}'.format(self._cfg.ds_name))
        with open('/app/data/output/{}_{}'.format(self._cfg.ds_name, self._cfg.output_file.replace('.h5', '.csv')), 'w') as out_file:
            for left_idx, left_label in enumerate(ids):
                for right_idx, right_label in enumerate(ids[left_idx:]):
                    out_file.write("{},{},{:0.3f}\n".format(left_label, right_label, node[left_idx, left_idx+right_idx]))
        f.close()

    def main(self):
        self.announcer("Starting conversion")
        self.convert()
        self.announcer("Loaded HDF5 file")