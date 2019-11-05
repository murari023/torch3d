class Metric(object):
    def reset(self):
        raise NotImplementedError

    def update(self, x, y):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError
