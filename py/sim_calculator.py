from py.utils import *
from functools import partial
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from py.configurator import Calculate

CHUNK_SIZE = 100


class SimCalculator(object):
    def __init__(self, config: Calculate, start_time):
        self.announcer = partial(announcer, process="Calculator", start=start_time)
        self.f = h5py.File('/app/data/output/{}'.format(config.output_file), 'r+')
        self.vectors = self.f['/vectors/{}'.format(config.ds_name)]
        self.sim = self.f.require_group("sim")
        self.ds = self.sim.create_dataset(config.ds_name,
                                          dtype=np.float32,
                                          shape=(len(self.vectors), len(self.vectors)),
                                          fillvalue=0.0,
                                          compression="gzip",
                                          compression_opts=9,
                                          shuffle=True)
        # self.CHUNK_COUNT = (((len(self.vectors) // CHUNK_SIZE) + 1) * ((len(self.vectors) // CHUNK_SIZE) + 2)) // 2

    def calculate_sims(self, left_min, left_max, right_min, right_max):
        sims = cosine_similarity(self.vectors[left_min:left_max, :], self.vectors[right_min:right_max])
        sims = np.nan_to_num(sims)
        if left_min == right_min:
            sims = np.triu(sims)
        self.ds[left_min:left_max, right_min:right_max] = sims

    def pair_iterator(self):
        chunks = [(c, min(c + CHUNK_SIZE, len(self.vectors))) for c in range(0, len(self.vectors), CHUNK_SIZE)]
        for left_index, left in enumerate(chunks):
            for right in chunks[left_index:]:
                yield left, right
            if left_index % 10 == 0:
                self.announcer("Chunk {:>3d}/{:>3d} completed".format(left_index, len(chunks)))

    def close(self):
        self.f.flush()
        self.f.close()

    def main(self):
        self.announcer("Started sim calculation task")
        for ((lm, lx), (rm, rx)) in self.pair_iterator():
            self.calculate_sims(lm, lx, rm, rx)
        self.announcer("finished calculating sims")
