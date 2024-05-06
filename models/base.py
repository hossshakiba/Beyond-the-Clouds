import os
from abc import abstractmethod

import torch


class BaseModel():
    def __init__(self, config, dataloader):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.config = config
        self.phase = config['phase']
        self.device = config['device']
        self.batch_size = self.config['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = config['n_epoch']
        self.lr = config['lr']
        self.dataloader = dataloader
        self.model_path = config[self.phase]['model_path']
        self.model_name = config[self.phase]['model_name']
        # self.metrics = metrics
        # self.schedulers = []
        # self.optimizers = []

    def train(self):
        print('Start Training...')
        self.train_step()

    def test(self):
        self.test_step()

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your model.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError(
            'You must specify how to do validation on your model.')

    def test_step(self):
        pass

    def save_model(self, model):
        """  Saves the model's state dictionary. """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        path = os.path.join(self.model_path, self.model_name)
        torch.save(model.state_dict(), path)
