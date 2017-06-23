from functools import partial
from py.configurator import Rotate
from py.utils import *

import numpy as np
from numpy.linalg import svd
from gensim.models import LsiModel


def varimax(phi, gamma=1, q=20, tol=1e-6):
    p, k = phi.shape
    r = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(phi, r)
        u, s, vh = svd(np.dot(phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        r = np.dot(u, vh)
        d = np.sum(s)
        if d/d_old < tol:
            break
    return np.dot(phi, r)


class Rotator(object):

    def __init__(self, config: Rotate, start_time):
        self._cfg = config
        self.announcer = partial(announcer, process="Rotator", start=start_time)

    def rotate(self):
        self.announcer("starting to rotate")
        model = LsiModel.load("/app/data/spaces/{}/lsi".format(self._cfg.space_name))
        self.announcer("loaded model")
        model.projection.u = varimax(model.projection.u)
        self.announcer("rotated u matrix")
        model.save("/app/data/spaces/{}/lsi_rotated".format(self._cfg.space_name))
        self.announcer("saved rotated matrix")

    def main(self):
        self.rotate()
