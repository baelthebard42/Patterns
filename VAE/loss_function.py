import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F




def sobel_edges(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    
    edge_x = torch.cat([F.conv2d(x[:, i:i+1, :, :], sobel_x, padding=1) for i in range(x.shape[1])], dim=1)
    edge_y = torch.cat([F.conv2d(x[:, i:i+1, :, :], sobel_y, padding=1) for i in range(x.shape[1])], dim=1)

    return torch.sqrt(edge_x**2 + edge_y**2)



def kullback_leibler(mu, log_var):
     return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

class vae_loss(nn.Module):
    def __init__(self, device):
        
        super(vae_loss, self).__init__()
        self.device=device
        
        

    def forward(self, y_target, y_pred, mu, log_var, beta=1):
       
        recon_loss = F.mse_loss(y_pred, y_target, reduction='mean').to(self.device)
        edge_loss = F.mse_loss(sobel_edges(y_pred), sobel_edges(y_target), reduction='mean').to(self.device)
        kl_loss =  kullback_leibler(mu, log_var)
        total_loss = recon_loss*0.3 + edge_loss*0.7 +  kl_loss*beta
        return  total_loss, recon_loss, edge_loss, kl_loss
