class BaseTrain(object):
    def __init__(self, model, data, val_data, config):
        self.model = model
        self.data = data
        self.validation = val_data
        self.config = config

    def train(self):
        raise NotImplementedError
