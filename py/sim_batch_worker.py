import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing as mp
import re
from functools import partial
import redis
from py.configurator import CalculateSettings


class SimBatchWorker(object):

    def __init__(self, cfg: CalculateSettings, announcer, batch_count, r: redis.StrictRedis=None):
        self._cfg = cfg
        self.use_redis = (self._cfg.top_n is not None) and (self._cfg.top_n > 0)
        self.process_id = int(re.search("(\d+)$", mp.current_process().name).group(1))
        self.announcer = partial(announcer, process="SimBatchWorker-%d" % self.process_id)
        self.batch_count = batch_count//1000
        if r:
            self.conn = r
        else:
            self.dest_file = "/app/data/temp_sim/sims-%d.csv" % self.process_id

    def write_to_file(self, lines):
        lt = "".join(["{},{},{:0.4f}\n".format(l, r, s) for l, r, s in lines])
        with open(self.dest_file, "a") as out_file:
            out_file.writelines(lt)

    def write_to_db(self, lines):
        sim_dict = {}
        for left, right, s in lines:
            sim = "{:0.4f}".format(s)
            try:
                sim_dict[left][right] = sim
            except KeyError:
                sim_dict[left] = {right: sim, }
        for left, right_sim in sim_dict.items():
            sim_array = np.array([(k, v) for k, v in right_sim.items()]).T
            keys = sim_array[0].astype(str)
            values = sim_array[1].astype(np.float32)
            part = np.argpartition(values, kth=-self._cfg.top_n)
            top_n = dict(zip(keys[part][-self._cfg.top_n:], values[part][-self._cfg.top_n:]))
            self.conn.hmset(left, top_n)


sbw: SimBatchWorker


def init_worker(cfg, announcer, batch_count, r=None):
    global sbw
    sbw = SimBatchWorker(cfg, announcer, batch_count, r)


def process_batch(data_a, data_b, batch_no):
    global sbw
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
    if sbw.use_redis:
        sbw.write_to_db(lines)
    else:
        sbw.write_to_file(lines)
    if batch_no % 1000 == 0:
        sbw.announcer(msg="Batch {:>4,d}k/{:,d}k".format(batch_no // 1000, sbw.batch_count))
