from torch.utils.data import Dataset
import torch
import os
from PIL import Image


class PatternDB(Dataset):

    def __init__(self, data_path, train=True, transform=None):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, index):
     path = os.path.join(self.data_path, f"{index}.jpg")
     image = Image.open(path).convert("RGB")

     if self.transform:
        image = self.transform(image)
     return image
    
    