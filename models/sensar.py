import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torcheval.metrics import PeakSignalNoiseRatio
import torchvision.models as models

from tqdm import tqdm
from PIL import Image
import numpy as np

from models.base import BaseModel


class SenSar(BaseModel):
    def __init__(self, network, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(SenSar, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)

    def set_input(self, data):
        ''' Sets the input data for the model. '''
        self.cloudy_images = data['input']['S2'][0][:, 2:5, :, :].to(self.device)
        self.sar = data['input']['S1'][0].to(self.device)
        self.cloud_free = data['target']['S2'][0][:, 2:5, :, :].to(self.device)

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
    
    def get_rgb(self, image):
        image = image.mul(0.5).add_(0.5)
        image = image.squeeze()
        image = image.mul(10000).add_(0.5).clamp_(0, 10000)
        image = image.permute(1, 2, 0).cpu().detach().numpy()
        image = image.astype(np.uint16)

        r = image[:,:,1]
        g = image[:,:,2]
        b = image[:,:,0]

        r = np.clip(r, 0, 10000)
        g = np.clip(g, 0, 10000)
        b = np.clip(b, 0, 10000)

        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)
        rgb = rgb.astype(np.uint8)

        return rgb
    
    def train_step(self):
        for epoch in range(self.epoch):
            overall_loss = 0
            dataloader_iter = tqdm(enumerate(self.dataloader), desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for i, train_data in dataloader_iter:
                self.set_input(train_data)

                self.optimizer.zero_grad()
                x_hat, mean, log_var = self.network(self.cloudy_images, self.sar)
                loss = self.loss_function(x_hat, self.cloud_free, mean, log_var, self.loss_func)

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            print(f"Epoch {epoch + 1}, Average Loss: {overall_loss / (i*self.batch_size):.5f}")
        self.save_model(self.network)
    
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
            x_hat, mean, log_var = self.network(self.cloudy_images, self.sar)
            loss = self.loss_function(x_hat, self.cloud_free, mean, log_var, self.loss_func)

            test_loss += loss.item()
            psnr.update(x_hat, self.cloud_free)

            for output_image, cloudy_image, cloud_free in zip(x_hat, self.cloudy_images, self.cloud_free):
                output_image = self.get_rgb(output_image)
                output_image = Image.fromarray(output_image)
                cloudy_image = self.get_rgb(cloudy_image)
                cloudy_image = Image.fromarray(cloudy_image)
                cloud_free = self.get_rgb(cloud_free)
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
