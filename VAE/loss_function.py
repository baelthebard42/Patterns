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
     log_var = torch.clamp(log_var, min=-10, max=10)
     return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

class vae_loss(nn.Module):
    def __init__(self, device):
        
        super(vae_loss, self).__init__()
        self.device=device
        
        

    def forward(self, y_target, y_pred, mu, log_var, beta=1):

        if torch.isnan(y_pred).any():
          print("NaN detected in reconstructed output, replacing with random constant")
          random_constant = torch.randn_like(y_pred) * 0.1  
          y_pred = torch.where(torch.isnan(y_pred), random_constant, y_pred)
        
        if torch.isinf(y_pred).any():
          print("Infinity detected in reconstructed output, replacing with random constant")
          random_constant = torch.randn_like(y_pred) * 0.1  
          y_pred = torch.where(torch.isinf(y_pred), random_constant, y_pred)

        assert not torch.isnan(y_target).any(), "y_target contains NaN values"
        assert not torch.isinf(y_target).any(), "y_target contains inf values"
       
        recon_loss = F.mse_loss(y_pred, y_target, reduction='mean').to(self.device)
        edge_loss = F.mse_loss(sobel_edges(y_pred), sobel_edges(y_target), reduction='mean').to(self.device)
        kl_loss =  kullback_leibler(mu, log_var)

        if torch.isnan(recon_loss):
         print("NaN detected in recon loss! Stopping training.")
         return None

        if torch.isnan(kl_loss):
         print("NaN detected in edge loss! Stopping training.")
         return None 

        if torch.isnan(edge_loss):
         print("NaN detected in edge loss! Stopping training.")
         return None 
    
        
        total_loss = recon_loss*0.8 + edge_loss*0.8 +  kl_loss*beta
        return  total_loss, recon_loss, edge_loss, kl_loss*beta
