from torch.utils.data import random_split
import pytorch_lightning as pl
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import ToTensor, Compose
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x).swapaxes(1, 3)
        self.y = np.array(y).swapaxes(1, 3)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index].astype('float32') / 255, self.y[index].astype('float32') / 255
        
class FishDataset(pl.LightningDataModule):
    def __init__(self, x_dir, y_dir, batch_size, num_workers):
        super().__init__()
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        x = []
        y = []
        for filename in sorted(os.listdir(self.x_dir)):
            img = cv2.imread(os.path.join(self.x_dir, filename))
            x.append(cv2.resize(img, (572, 572)))
        for filename in sorted(os.listdir(self.y_dir)):
            img = cv2.imread(os.path.join(self.y_dir, filename), cv2.IMREAD_GRAYSCALE)
            y.append(np.expand_dims(cv2.resize(img, (388, 388)), axis=2))
        
        trainx = x[:900]
        trainy = y[:900]

        valx = x[-900:]
        valy = y[-900:]

        self.train = CustomDataset(x = trainx, y = trainy)
        self.val = CustomDataset(x = valx, y = valy)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

