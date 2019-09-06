from DensePS import DenseParameterServer
from SparsePS import SparseParameterServer
import ray
from torch import FloatTensor
import timeit
import numpy as np

N_elems = 50
N = 1000

# TODO: can be extended for benchmarking


def dense_main_loop(dense_ps1):
    for i in range(N_elems):
        v_id = dense_ps1.pull.remote([i])
        v = ray.get(v_id)[0]
        assert(v[0][0] == i)


def dense_test():
    keys = [k for k in range(N_elems)]
    values = [FloatTensor(np.ones((N, N)).astype(np.float) * v)
              for v in range(N_elems)]

    dense_ps1 = DenseParameterServer.remote(keys, values)
    dense_main_loop(dense_ps1)


def sparse_test():
    N_rows = 100
    keys = ["table1"] * N_rows
    indices = [i for i in range(N_rows)]
    values = [FloatTensor(np.ones((1, N)).astype(np.float) * v)
              for v in range(N_rows)]
    sparse_ps = SparseParameterServer.remote(keys, indices, values)
    for i in range(N_rows):
        v_id = sparse_ps.pull.remote(["table1"], [i])
        v = ray.get(v_id)[0]
        assert(v.shape == (1, N))

    id = sparse_ps.get_keys.remote()
    keys = ray.get(id)
    assert(keys == ["table1"])


if __name__ == "__main__":
    ray.init()
    dense_test()
    sparse_test()
