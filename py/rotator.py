from functools import partial

from gensim.models.lsimodel import Projection

from py.configurator import Rotate
from py.utils import *


class Rotator(object):
    def __init__(self, config: Rotate, start_time: datetime):
        self._cfg = config
        self.announcer = partial(announcer, process="Creator", start=start_time)
        self._load_projection()

    def _load_projection(self):
        proj = Projection.load("/app/data/spaces/{}/lsi.projection".format(self._cfg.space_name))
        self.announcer("loaded projection")
        self.u = proj.u
        self.k = proj.k

    def main(self):
        rotatated_u, rot_mat = varimax(self.u[:, :min(self._cfg.num_dims, self.k)])
        self.announcer("rotated space")
        np.save("/app/data/spaces/{}/rotmat_{}.npy".format(self._cfg.space_name, self._cfg.num_dims), rot_mat)
        self.announcer("saved rotation")
