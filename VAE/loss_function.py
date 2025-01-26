import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 14, 21]):  
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        
    def forward(self, x, y):

        loss = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += F.mse_loss(x, y)
                
        return loss



def kullback_leibler(mu, log_var):
     return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

class vae_loss(nn.Module):
    def __init__(self):
        super(vae_loss, self).__init__()
        self.perceptual = PerceptualLoss()

    def forward(self, y_target, y_pred, mu, log_var, beta):
        perceptual_loss = (1-beta)*self.perceptual(y_target, y_pred)
        kl_loss = beta * kullback_leibler(mu, log_var)
        return  perceptual_loss+kl_loss, perceptual_loss, kl_loss
