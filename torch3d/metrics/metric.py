class Metric(object):
    def reset(self):
        raise NotImplementedError

    def update(self, output, target):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
