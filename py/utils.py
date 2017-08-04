import subprocess
from datetime import datetime
from enum import Enum

import numpy as np
from mpi4py import MPI

TASK_TYPE = Enum('TASK_TYPE', 'CREATE PROJECT CALCULATE ROTATE')
PAIR_MODE = Enum('PAIR_MODE', 'ALL CROSS LIST')
OUTPUT_FORMAT = Enum('OUTPUT_FORMAT', 'H5 CSV')
DISTANCE_METRIC = Enum('DISTANCE_METRIC', 'COSINE ABS_DIFFERENCE R')


def run_cmd(cmd, raw=False):
    if raw:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def announcer(msg, process, start):
    diff = datetime.now()-start
    days = diff.days
    hours = diff.seconds//3600
    minutes = diff.seconds//60 - hours * 60
    seconds = diff.seconds - hours * 3600 - minutes * 60
    print("{process:<20}{ts:<20}{msg:<39}".format(process=process,
                                                  ts="{:02d}d {:02d}h {:02d}m {:02d}s".format(days,
                                                                                              hours,
                                                                                              minutes,
                                                                                              seconds),
                                                  msg=msg))


def varimax(mat: np.matrix, kaiser_norm: bool = True, eps: float = 1e-5) -> (np.matrix, np.matrix):
    """
    :param mat:
    :param kaiser_norm:
    :param eps:
    :returns: tuple (loadings, rotmat)
        WHERE
        np.matrix loadings are the loadings in the rotated space
        np.matrix rotmat is the rotation matrix
    :rtype: (np.matrix, np.matrix)
    """
    from numpy.linalg import svd
    x = mat.copy()
    p, nc = x.shape
    if nc < 2:
        return x
    if kaiser_norm:
        sc = np.sqrt(np.sum(np.square(x), axis=1))
        x = np.divide(x, sc[:, np.newaxis])
    tt = np.eye(nc)
    d = 0.0
    for i in range(1000):
        print("starting loop {}".format(i))
        z = np.dot(x, tt)
        z_cubed = np.power(z, 3)
        sum_squares = np.sum(np.square(z), axis=0)
        ss_over_p = sum_squares / p
        diag = np.diag(np.ravel(ss_over_p))
        b = np.dot(x.T, (z_cubed - np.dot(z, diag)))
        u, s, vt = svd(b)
        tt = np.dot(u, vt)
        d_past = d
        d = np.sum(s)
        if d < d_past * (1 + eps):
            break
    z = np.dot(x, tt)
    if kaiser_norm:
        z = np.multiply(x, sc[:, np.newaxis])
    return z, tt


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = "/app/data/profile/{}.{}".format(filename, comm.rank)
                pr.dump_stats(filename_r)

            return result

        return wrap_f

    return prof_decorator
