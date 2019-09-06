import ray

# Dense parameter server
# Granularity is the parameter


@ray.remote
class DenseParameterServer(object):
    def __init__(self, keys, values):
        values = [value.clone() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self, keys):
        return [self.weights[key] for key in keys]
