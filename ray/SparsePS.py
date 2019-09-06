import ray

# Sparse parameter server
# Granularity is the slice of the first dimension of the tensor
# To address a parameter we need to pass the index of the leading
# dimension. e.g. row of an embedding matrix
@ray.remote
class SparseParameterServer(object):
    def __init__(self, keys, indices, values):
        values = [value.clone() for value in values]
        self.weights = {}
        for k, i, v in zip(keys, indices, values):
            if k not in self.weights:
                self.weights[k] = {}
            self.weights[k][i] = v

    def push(self, keys, indices, values):
        for key, id, value in zip(keys, idx,  values):
            self.weights[key][id] = value

    def pull(self, keys, indices):
        return [self.weights[key][idx] for key, idx in zip(keys, indices)]

    def pull_dense(self, keys):
        ret = {}
        for k in keys:
            ret[k] = [(i, v) for i, v in self.weights[k].items()]
        return ret

    def get_keys(self):
        print(list(self.weights.keys()))
        return list(self.weights.keys())
