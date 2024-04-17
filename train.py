import torch
import torch.nn as nn
from torch.optim import Adam

from models.network import VAE
from util import get_data

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 32
x_dim = 256
hidden_dim = 400
latent_dim = 200
lr = 1e-3
epochs = 20

dataloader = get_data(256, batch_size)

model = VAE().to(DEVICE)


MSE = nn.MSELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = MSE(x_hat, x)
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for i, (cloudy, cloud_free, _) in enumerate(dataloader):
        cloud_free = cloud_free.to(DEVICE)
        x = cloudy[0][:, :3, :, :].to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x_hat, cloud_free, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("\tEpoch", epoch + 1, "complete!",
          "\tAverage Loss: ", overall_loss / (i*batch_size))

# Evaluation
model.eval()

with torch.no_grad():
    for i, (cloudy, cloud_free, _) in enumerate(dataloader):
        cloud_free = cloud_free.to(DEVICE)
        cloudy_x0 = cloudy[0][:, :3, :, :].to(DEVICE)
        x_hat, _, _ = model(cloud_free)

        break
