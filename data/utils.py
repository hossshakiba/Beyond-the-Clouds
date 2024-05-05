from torch.utils.data import DataLoader
import numpy as np

from .dataset import Sen2_MTC

def get_rgb(image):
    image = image.mul(0.5).add_(0.5)
    image = image.squeeze()
    image = image.mul(10000).add_(0.5).clamp_(0, 10000)
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = image.astype(np.uint16)

    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]

    r = np.clip(r, 0, 2000)
    g = np.clip(g, 0, 2000)
    b = np.clip(b, 0, 2000)

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

def get_data(path, batch, mode='train'):
    dataset = Sen2_MTC(path, mode)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)
    return dataloader