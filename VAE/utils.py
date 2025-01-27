import torch
import numpy as np
import matplotlib.pyplot as plt


def denormalize(tensor):
   
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean 

def show_reconstructed_images(original, reconstructed, num_images=8):
  
    original = denormalize(original).cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = torch.sigmoid(reconstructed)  
    reconstructed = denormalize(reconstructed).cpu().permute(0, 2, 3, 1).numpy()

    num_images = min(num_images, original.shape[0])

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        axes[0, i].imshow(np.clip(original[i], 0, 1))  
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(reconstructed[i], 0, 1))
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.show()


def display_images(model, dataloader, device):
    model.eval() 
    with torch.no_grad():
       
        data_batch = next(iter(dataloader)).to(device)

      
        mu, log_var = model.encode(data_batch)
        z = model._reparameterize(mu, log_var)
        reconstructed = model.decode(z)

       
        show_reconstructed_images(data_batch, reconstructed)





