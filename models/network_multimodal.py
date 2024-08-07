import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from models.ViTs.deepvit import DeepViT




# --- Channel Attention (CA) Layer --- #
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attn = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_attn(y)
        return x * y 


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.spatial_attn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        h = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_attn(h)

        return x * y


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 768)

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64)
        )
        self.up1 = nn.ConvTranspose2d(in_channels=64,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(67, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PReLU(512)
        )
        self.up2 = nn.ConvTranspose2d(in_channels=512,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(515, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(128)
        )
        self.up3 = nn.ConvTranspose2d(in_channels=128,
                                      out_channels=128,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(131, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(32)
        )
        self.up4 = nn.ConvTranspose2d(in_channels=32,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(35, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
            )
            
        self.attn_layer1 = CALayer(channel=64)
        self.attn_layer2 = CALayer(channel=512)
        self.attn_layer3 = CALayer(channel=128)
        self.attn_layer4 = CALayer(channel=32)

        self.tanh = nn.Tanh()
        

    def forward(self, img, mean, var):
        _, _, H, W = img.size()
        std = torch.exp(var / 2)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # decode
        x = self.fc1(z)
        x = self.fc2(x)
        x = x.view(-1, 3, 16, 16)
        hidden_map = x
        condition = F.interpolate(img, size=(H//16, W//16), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer1(x)
        x = self.attn_layer1(x)
        x = self.up1(x)
        condition = F.interpolate(img, size=(H//8, W//8), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer2(x)
        x = self.attn_layer2(x)
        x = self.up2(x)
        condition = F.interpolate(img, size=(H//4, W//4), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer3(x)
        x = self.attn_layer3(x)
        x = self.up3(x)
        condition = F.interpolate(img, size=(H//2, W//2), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer4(x)
        x = self.attn_layer4(x)
        x = self.up4(x)
        x = torch.cat((x, img), 1)

        output = self.layer5(x)
        output = self.tanh(output)
        
        return output, hidden_map
    
class ResNet50(nn.Module):
    def __init__(self, num_classes, num_dim):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # self.norm = nn.BatchNorm1d(num_classes)
        self.norm = nn.LayerNorm(num_classes)
        

        self.mu = nn.Linear(num_classes,
                            num_dim)
        self.log_var = nn.Linear(num_classes,
                                 num_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = self.norm(x)
        mu = self.mu(x)  # take only mu_tokens
        log_var = self.log_var(x)  # take only log_var_tokens

        return mu, log_var

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepViT(
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 512,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1,
                        num_dim=50
                    )
        self.resnet = ResNet50(num_classes = 512,
                               num_dim=50)

        self.decoder = Decoder()
         
    def en(self, x):
        return self.encoder(x)
    
    def de(self, x, mean, log_var):
        return self.decoder(x, mean, log_var)
        
    def forward(self, x, sar):
        mean_x, log_var_x = self.en(x)
        mean_sar, log_var_sar = self.resnet(sar)
        mean, log_var = torch.cat((mean_x, mean_sar), 1), torch.cat((log_var_x, log_var_sar), 1)
        output, hidden_map = self.de(x, mean, log_var)
        output = torch.add(x, output)

        return output, mean, log_var
