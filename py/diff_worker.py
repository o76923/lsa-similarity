from functools import partial

import h5py as h5
import numpy as np
from mpi4py import MPI

CHUNK_SIZE = 100


class DiffWorker(object):
    def __init__(self):
        self.parent_comm = MPI.Comm.Get_parent().Merge()
        self.world_comm = MPI.COMM_WORLD
        self.rank = self.parent_comm.Get_rank()
        self.size = self.parent_comm.Get_size()
        self._listen()
        self._init_chunks()

    def _listen(self):
        parent_announcer = self.parent_comm.bcast(None)
        self.announcer = partial(parent_announcer, process="Worker-{}".format(self.rank))
        self.ds_name = self.parent_comm.bcast(None)
        self.vectors = self.parent_comm.bcast(None)
        file_name = self.parent_comm.bcast(None)
        self.f = h5.File(file_name, 'a', driver='mpio', comm=self.world_comm, libver='latest')
        self.diff = self.f['/difference/{}'.format(self.ds_name)]

    def _init_chunks(self):
        self.chunks = [self.vectors[n:n + CHUNK_SIZE] for n in range(0, len(self.vectors), CHUNK_SIZE)]
        self.chunk_count = ((len(self.chunks) + 1) * (len(self.chunks))) // 2 - 1

    def pair_iterator(self):
        vector_length = len(self.vectors)
        left_chunk_count = vector_length // 10
        for left_index in range(0, left_chunk_count, self.size - 1):
            yield self.vectors[left_index * (10 * 10) + self.rank - 1:(left_index + 1) * (10 * 10) + self.rank - 1]
            # i = 0
            # for left_index, left_chunk in enumerate(self.chunks):
            #     left_min = left_index*CHUNK_SIZE
            #     left_max = (left_index+1)*CHUNK_SIZE
            #     for right_index, right_chunk in enumerate(self.chunks[left_index:], left_index):
            #         right_min = right_index*CHUNK_SIZE
            #         right_max = (right_index+1)*CHUNK_SIZE
            #         i += 1
            #         yield i, left_min, left_max, right_min, right_max, left_chunk, right_chunk

    def close(self):
        self.f.flush()
        self.f.close()
        self.parent_comm.Disconnect()

    @staticmethod
    def subtract(lc, rc):
        return np.abs(lc - rc[:, np.newaxis])

    def triangle_swap(self, d):
        return np.swapaxes(np.triu(np.swapaxes(d, 0, 2), k=self.rank - 1), 0, 1)

    @staticmethod
    def regular_swap(d):
        return np.swapaxes(np.swapaxes(d, 1, 2), 0, 2)

    def assign(self, lm, lx, rm, rx, d):
        self.diff[lm:lx, :, rm:rx] = d

    def assign2(self, right_min, d):
        self.diff[self.rank - 1::self.size, :, right_min:right_min + CHUNK_SIZE] = d

    def calculate(self):
        try:
            for i, lm, lx, rm, rx, lc, rc in self.pair_iterator()[self.rank - 1::self.size]:
                d = self.subtract(lc, rc)
                if lm == rm:
                    d = self.triangle_swap(d)
                else:
                    d = self.regular_swap(d)
                self.assign(lm, lx, rm, rx, d)
                self.announcer(msg="Chunk {:>3d}/{:>3d} completed".format(i, self.chunk_count))
        finally:
            self.close()

    def calculate3(self):
        vs = np.array_split(self.vectors, self.size - 1)
        vs_length = [len(v) for v in vs]
        left_chunk = vs[self.rank - 1]
        for right_index, right_chunk in enumerate(vs):
            # if self.rank - 1 >= right_index:
            d = self.subtract(left_chunk, right_chunk)
            # if right_index == self.rank - 1:
            #     d = self.triangle_swap(d)
            # else:
            d = self.regular_swap(d)
            # else:
            #     d = np.zeros(shape=(left_chunk.shape[0], left_chunk.shape[0][0], right_chunk.shape[0]))
            with self.diff.collective:
                self.diff[sum(vs_length[:self.rank - 1]):sum(vs_length[:self.rank]), :,
                sum(vs_length[:right_index]):sum(vs_length[:right_index + 1])] = d
                self.f.flush()
            self.announcer("finished chunk {}/{}".format(right_index, self.size - 1))

            # @profile("calculate2")
            # def calculate2(self):
            #     try:
            #         lc = self.vectors[self.rank-1::self.size-1]
            #         with self.diff.collective:
            #             for left_index, left_chunk in enumerate(np.array_split(lc, CHUNK_SIZE)):
            #                 d = self.subtract(left_chunk, self.vectors)
            #                 print(d.shape)
            #                 d = self.regular_swap(d)
            #                 # print(d.shape)
            #                 for i in range(len(d)):
            #                     d[i, :, :i*(self.size - 1)+self.rank-1] = 0.0
            #                 print(d.shape)
            #                 left_min = left_index * (CHUNK_SIZE * (self.size - 1)) + self.rank - 1
            #                 left_max = (left_index+1) * (CHUNK_SIZE * (self.size - 1)) + self.rank - 1
            #                 left_incr = self.size - 1
            #                 # print(self.diff.shape)
            #                 print(self.diff[left_min:left_max].shape)
            #                 print(self.diff[left_min:left_max:left_incr].shape)
            #                 # self.diff[left_min:left_max:left_incr] = d
            #                 self.f.flush()
            #                 self.announcer(msg="Subtraction done for chunk {}".format(left_index))
            #     finally:
            #         self.close()


if __name__ == "__main__":
    dw = DiffWorker()
    try:
        dw.calculate3()
    finally:
        dw.close()
