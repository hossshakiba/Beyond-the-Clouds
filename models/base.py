from abc import abstractmethod


class BaseModel():
    def __init__(self, config, train_dataloader):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.config = config
        self.phase = config['phase']
        self.device = config['device']
        self.batch_size = self.config['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = config['n_epoch']
        self.lr = config['lr']
        self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        # self.metrics = metrics
        # self.schedulers = []
        # self.optimizers = []

    def train(self):
        print('Start Training...')
        self.train_step()

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your model.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError(
            'You must specify how to do validation on your model.')

    def test_step(self):
        pass

    def save_model(self, model, path):
        pass
