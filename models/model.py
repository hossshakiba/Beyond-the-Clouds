import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from models.base import BaseModel


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)

    def set_input(self, data):
        ''' Sets the input data for the model. '''
        self.cloudy_images = [image.to(self.device) for image in data.get('cloudy_images')]
        self.cloud_free = data.get('cloud_free').to(self.device)
        self.path = data['path']
        self.batch_size = len(data['path'])

    def loss_function(self, x, x_hat, mean, log_var):
        MSE = nn.MSELoss()
        reproduction_loss = MSE(x_hat, x)
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    def train_step(self):
        for epoch in range(self.epoch):
            overall_loss = 0
            dataloader_iter = tqdm(enumerate(self.train_dataloader), desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.train_dataloader))
            for i, train_data in dataloader_iter:
                self.set_input(train_data)

                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.network(self.cloudy_images[0])
                loss = self.loss_function(x_hat, self.cloud_free, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            print("\tEpoch", epoch + 1,"\tAverage Loss: ", overall_loss / (i*self.batch_size))