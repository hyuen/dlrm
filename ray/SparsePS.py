import ray

# Sparse parameter server
# Granularity is the slice of the first dimension of the tensor
# To address a parameter we need to pass the index of the leading
# dimension. e.g. row of an embedding matrix
@ray.remote
class SparseParameterServer(object):
    def __init__(self, keys, indices, values):
        print("a")
        values = [value.clone() for value in values]
        self.weights = {}
        for k, i, v in zip(keys, indices, values):
            if k not in self.weights:
                self.weights[k] = {}
            self.weights[k][i] = v
                
        #self.weights = dict(zip(keys, indices, values))

    #def __init__(self, keys, values):
    #    print("b")
    #    #assert(0)
        
    #def push_dense(self, keys, values):
    #    pass
        
    #def push(self, keys, indices, values):
    #    for key, id, value in zip(keys, idx,  values):
    #        self.weights[key, id] += value

    def pull(self, keys, indices):
        return [self.weights[key][idx] for key, idx in zip(keys, indices)]

    #def pull_dense(self, keys):
    #    pass
