import sys
from functools import partial
from itertools import combinations

import h5py
from sklearn.metrics.pairwise import cosine_similarity

from py.configurator import Calculate
from py.utils import *

CHUNK_SIZE = 100
world_comm = MPI.COMM_WORLD


class SimCalculator(object):
    def __init__(self, config: Calculate, start_time):
        self.announcer = partial(announcer, process="Calculator", start=start_time)
        self._cfg = config
        if config.output_format == OUTPUT_FORMAT.H5:
            self.f = h5py.File('/app/data/{}'.format(self._cfg.output_file), 'r+')
        if config.output_format == OUTPUT_FORMAT.CSV:
            self.f = h5py.File('/app/data/{}'.format(self._cfg.output_file)[:-3] + 'h5', 'r+')
        self.vectors = self.f['/vectors/{}'.format(self._cfg.ds_name)]
        if self._cfg.distance_metric == DISTANCE_METRIC.COSINE:
            self.sim = self.f.require_group("sim")
            sim_shape = (len(self.vectors), len(self.vectors))
            self.ds = self.sim.require_dataset(self._cfg.ds_name,
                                               dtype='float16',
                                               shape=sim_shape,
                                               fillvalue=0.0,
                                               compression="gzip",
                                               compression_opts=9,
                                               shuffle=True)
        elif self._cfg.distance_metric == DISTANCE_METRIC.ABS_DIFFERENCE:
            self.sim = self.f.require_group("difference")
            sim_shape = (len(self.vectors), len(self.vectors[0]), len(self.vectors))
            self.ds = self.sim.require_dataset(self._cfg.ds_name,
                                               dtype='float16',
                                               shape=sim_shape,
                                               fillvalue=0.0,
                                               chunks=(CHUNK_SIZE, len(self.vectors[0]), CHUNK_SIZE),
                                               compression="gzip",
                                               compression_opts=9,
                                               shuffle=True,
                                               )
            self.f.flush()

    def calculate_cosine_sims(self, left_min, left_max, right_min, right_max):
        sims = cosine_similarity(self.vectors[left_min:left_max], self.vectors[right_min:right_max])
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

    def convert_to_csv(self):
        sims = self.ds[:]
        with open('/app/data/{}'.format(self._cfg.output_file), "w") as out_file:
            for (left, right), val in zip(combinations(self.f["/input/id"], 2), sims[np.triu_indices_from(sims, k=1)]):
                out_file.write("{},{},{:0.3f}\n".format(left, right, val))

    def launch_diff_workers(self, v):
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=['-m', 'py.diff_worker'],
                                   maxprocs=self._cfg.num_cores).Merge()
        # maxprocs=2).Merge()
        self.announcer("Workers launched")
        comm.bcast(self.announcer)
        comm.bcast(self._cfg.ds_name)
        comm.bcast(v)
        file_name = self.f.filename
        comm.bcast(file_name)
        self.announcer("Broadcast filename and announcer")
        comm.Disconnect()
        self.announcer("Disconnected from COMM_SELF")

    def main(self):
        self.announcer("Started sim calculation task")
        if self._cfg.distance_metric == DISTANCE_METRIC.COSINE:
            for ((lm, lx), (rm, rx)) in self.pair_iterator():
                self.calculate_cosine_sims(lm, lx, rm, rx)
        elif self._cfg.distance_metric == DISTANCE_METRIC.ABS_DIFFERENCE:
            v = np.vstack(self.vectors)
            self.announcer("made vstack")
            vs = np.array_split(v, 10)
            vs_length = [len(v) for v in vs]
            for left_index in range(len(vs)):
                left_min = sum(vs_length[:left_index])
                left_max = sum(vs_length[:left_index + 1])
                left_chunk = vs[left_index]
                for right_index in range(left_index, len(vs)):
                    right_min = sum(vs_length[:right_index])
                    right_max = sum(vs_length[:right_index + 1])
                    right_chunk = vs[right_index]
                    d = np.abs(left_chunk - right_chunk[:, np.newaxis])
                    if left_index == right_index:
                        d = np.swapaxes(d, 0, 2)
                        d = np.triu(d)
                        d = np.swapaxes(d, 0, 1)
                    else:
                        d = np.swapaxes(d, 1, 2)
                        d = np.swapaxes(d, 0, 2)
                    # self.announcer("calculated {}, {}".format(left_index, right_index))
                    self.ds[left_min:left_max, :, right_min:right_max] = d
                    self.f.flush()
                    self.announcer(
                            "{}, {}, {:> 7,d}:{:> 7,d}, {:> 7,d}:{:> 7,d}".format(left_index, right_index, left_min,
                                                                                  left_max, right_min, right_max))
        self.announcer("finished calculating sims")
        if self._cfg.output_format == OUTPUT_FORMAT.CSV:
            self.announcer("converting to CSV")
            self.convert_to_csv()
            self.announcer("finished CSV conversion")
        self.f.close()
