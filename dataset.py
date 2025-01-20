from torch.utils.data import Dataset
import torch
import os
from PIL import Image

START = 10548

class PatternDB(Dataset):

    def __init__(self, data_path, train=True, transform=None):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def _getitem__(self, index):
     path = os.path.join(self.data_path, f"{START+index}.png")
     image = Image.open(path).convert("RGB")

     if self.transform:
        image = self.transform(image)
     return image