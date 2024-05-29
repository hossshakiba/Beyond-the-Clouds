import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torcheval.metrics import PeakSignalNoiseRatio
import torchvision.models as models

from tqdm import tqdm
from PIL import Image

from models.base import BaseModel
from data.utils import get_rgb


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

    def loss_function(self, x, x_hat, mean, log_var, loss_type):
        MSE = nn.MSELoss()
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if loss_type["name"] == "default":
            reproduction_loss = MSE(x_hat, x)
            return reproduction_loss + KLD
        elif loss_type["name"] == "vgg":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].to(self.device)
            perceptual_loss_weight = loss_type["args"]["loss_weight"]
            reproduction_loss = MSE(vgg(x_hat), vgg(x))
            return perceptual_loss_weight*reproduction_loss + KLD

    def train_step(self):
        best_overall_loss = float('inf')
        for epoch in range(self.epoch):
            overall_loss = 0
            dataloader_iter = tqdm(enumerate(self.dataloader), desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for i, train_data in dataloader_iter:
                self.set_input(train_data)

                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.network(self.cloudy_images[0])
                loss = self.loss_function(x_hat, self.cloud_free, mean, log_var, self.loss_func)

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            if overall_loss < best_overall_loss:
                best_overall_loss = overall_loss
                self.save_model(self.network)
                
            print(f"Epoch {epoch + 1}, Average Loss: {overall_loss / (i*self.batch_size):.5f}")

    
    def test_step(self):
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

        psnr = PeakSignalNoiseRatio().to(self.device)

        counter = 0
        test_loss = 0.0
        test_psnr = 0.0

        dataloader_iter = tqdm(enumerate(self.dataloader), desc=f'Testing...', total=len(self.dataloader))
        for _, test_data in dataloader_iter:
            self.set_input(test_data)
            x_hat, mean, log_var = self.network(self.cloudy_images[0])
            loss = self.loss_function(x_hat, self.cloud_free, mean, log_var, self.loss_func)

            test_loss += loss.item()
            psnr.update(x_hat, self.cloud_free)

            for output_image, cloudy_image, cloud_free in zip(x_hat, self.cloudy_images[0], self.cloud_free):
                output_image = get_rgb(output_image)
                output_image = Image.fromarray(output_image)
                cloudy_image = get_rgb(cloudy_image)
                cloudy_image = Image.fromarray(cloudy_image)
                cloud_free = get_rgb(cloud_free)
                cloud_free = Image.fromarray(cloud_free)
                
                model_results_path = os.path.join("model_outputs", self.model_name.split('.')[0], "results")
                os.makedirs(model_results_path, exist_ok=True)
                os.makedirs(f"{model_results_path}/outputs", exist_ok=True)
                os.makedirs(f"{model_results_path}/inputs", exist_ok=True)
                os.makedirs(f"{model_results_path}/ground_truth", exist_ok=True)

                output_path = os.path.join(f"{model_results_path}/outputs/{counter}.png")
                input_path = os.path.join(f"{model_results_path}/inputs/{counter}.png")
                gt_path = os.path.join(f"{model_results_path}/ground_truth/{counter}.png")

                output_image.save(output_path)
                cloudy_image.save(input_path)
                cloud_free.save(gt_path)
                counter += 1

            test_loss = test_loss / len(self.dataloader)
            test_psnr = psnr.compute()

        print(f"Loss: {test_loss:.5f}, PSNR: {test_psnr:.5f}")
