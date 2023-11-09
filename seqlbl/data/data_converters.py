class AbstractDataConverter(object):
    def __init__(self):
        pass

    def convert(self, batch):
        raise NotImplementedError
