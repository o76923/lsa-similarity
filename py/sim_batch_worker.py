import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing as mp
import re
from functools import partial


class SimBatchWorker(object):

    def __init__(self, announcer, batch_count):
        self.process_id = int(re.search("(\d+)$", mp.current_process().name).group(1))
        self.dest_file = "/app/data/temp_sim/sims-%d.csv" % self.process_id
        self.announcer = partial(announcer, process="SimBatchWorker-%d" % self.process_id)
        self.batch_count = batch_count//1000

    def write_to_file(self, lines):
        lt = "".join(["{},{},{:0.4f}\n".format(l, r, s) for l, r, s in lines])
        with open(self.dest_file, "a") as out_file:
            out_file.writelines(lt)

_sbw: SimBatchWorker


def init_worker(announcer, batch_count):
    global _sbw
    _sbw = SimBatchWorker(announcer, batch_count)


def process_batch(data_a, data_b, batch_no):
    global _sbw
    keys_a, mat_a = data_a
    keys_b, mat_b = data_b
    sims = cdist(mat_a, mat_b, 'cosine')
    sims = np.ones(shape=sims.shape, dtype=np.float) - sims
    sim_iterator = sims.flat
    lines = []
    if keys_a == keys_b:
        for l in keys_a:
            for r in keys_b:
                sim = sim_iterator.__next__()
                if l < r:
                    lines.append((l, r, sim))
    else:
        for l in keys_a:
            for r in keys_b:
                sim = sim_iterator.__next__()
                lines.append((l, r, sim))
    _sbw.write_to_file(lines)
    if batch_no % 1000 == 0:
        _sbw.announcer(msg="Batch {:>4,d}k/{:,d}k".format(batch_no//1000, _sbw.batch_count))
