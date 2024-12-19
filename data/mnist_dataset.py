import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, device, csv_file, use_top=-1, transform=None):
        self.data = pd.read_csv(csv_file)
        if use_top > -1:
            self.data = self.data.head(use_top)
        self.transform = transform
        
        self.device=device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the first column is the label and the rest are pixel values
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        image = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float).reshape(1, 28, 28)
        
        # normalize to [-1,1] for stability
        image /= 255.0
        image = (image - 0.5) * 2
        
        if self.transform:
            image = self.transform(image)
        
        image = image.to(device=self.device)
        label = label.to(device=self.device)
        return image, label

