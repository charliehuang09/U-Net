import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from torchvision.transforms import CenterCrop
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Unet(pl.LightningModule):
    def __init__(self, channels, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.epoch = 1
        self.loss_fn = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros(1, 3, 572, 572)
        #activation
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.save_hyperparameters()

        #first block
        self.conv1 = nn.Conv2d(channels, 64, (3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))
        #first max pool
        self.maxpool3 = nn.MaxPool2d((2,2))

        #second block
        self.conv4 = nn.Conv2d(64, 128, (3,3))
        self.conv5 = nn.Conv2d(128, 128, (3,3))
        #second max pool
        self.maxpool6 = nn.MaxPool2d((2,2))

        #third block
        self.conv7 = nn.Conv2d(128, 256, (3,3))
        self.conv8 = nn.Conv2d(256, 256, (3,3))
        #third max pool
        self.maxpool9 = nn.MaxPool2d((2,2))

        #forth block
        self.conv10 = nn.Conv2d(256, 512, (3, 3))
        self.conv11 = nn.Conv2d(512, 512, (3,3))
        #forth max pool
        self.maxpool12 = nn.MaxPool2d((2,2))

        #-------------------------------------------

        self.conv13 = nn.Conv2d(512, 1024, (3,3))
        self.conv14 = nn.Conv2d(1024, 1024, (3,3))
        
        #-------------------------------------------

        #first block
        self.up15 = nn.ConvTranspose2d(1024, 512, (2,2), 2)
        #cat
        self.conv16 = nn.Conv2d(1024, 512, (3,3))
        self.conv17 = nn.Conv2d(512, 512, (3,3))

        #second block
        self.up18 = nn.ConvTranspose2d(512, 256, (2,2), 2)
        #cat
        self.conv19 = nn.Conv2d(512, 256, (3,3))
        self.conv20 = nn.Conv2d(256, 256, (3,3))

        #third block
        self.up21 = nn.ConvTranspose2d(256, 128, (2,2), 2)
        #cat
        self.conv22 = nn.Conv2d(256, 128, (3,3))
        self.conv23 = nn.Conv2d(128, 128, (3,3))

        #forth block
        self.up24 = nn.ConvTranspose2d(128, 64, (2,2), 2)
        #cat
        self.conv25 = nn.Conv2d(128, 64, (3,3))
        self.conv26 = nn.Conv2d(64, 64, (3,3))
        self.conv27 = nn.Conv2d(64, 1, (1,1))
    
    def forward(self, x):
        #first block
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x1 = x
        x = self.maxpool3(x)

        #second block
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x2 = x
        x = self.maxpool6(x)

        #third block
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x3 = x
        x = self.maxpool9(x)

        #forth block
        x = self.conv10(x)
        x = self.activation(x)
        x = self.conv11(x)
        x = self.activation(x)
        x4 = x
        x = self.maxpool12(x)

        #-------------------------------------------
        x = self.conv13(x)
        x = self.activation(x)
        x = self.conv14(x)
        x = self.activation(x)

        #-------------------------------------------

        #first block
        x = self.up15(x)
        crop = CenterCrop((x.shape[2], x.shape[2]))
        x4 = crop(x4)
        x = torch.cat([x4, x], dim=1)
        x = self.conv16(x)
        x = self.activation(x)
        x = self.conv17(x)
        x = self.activation(x)

        #second block
        x = self.up18(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x3 = crop(x3)
        x = torch.cat([x3, x], dim=1) 
        x = self.conv19(x)
        x = self.activation(x)
        x = self.conv20(x)
        x = self.activation(x)

        #third block
        x = self.up21(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x2 = crop(x2)
        x = torch.cat([x2, x], dim=1)
        x = self.conv22(x)
        x = self.activation(x)
        x = self.conv23(x)
        x = self.activation(x)
        
        #forth block
        x = self.up24(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x1 = crop(x1)
        x = torch.cat([x1, x], dim=1)
        x = self.conv25(x)
        x = self.activation(x)
        x = self.conv26(x)
        self.activation(x)
        x = self.conv27(x)
        x = self.sigmoid(x)
        # x = self.activation(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(y))
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(torch.from_numpy(x[:8].to('cpu').numpy().astype(np.float32)))
            self.logger.experiment.add_image("X Images", grid, self.epoch)

            grid = torchvision.utils.make_grid(torch.from_numpy(y[:8].to('cpu').numpy().astype(np.float32)))
            self.logger.experiment.add_image("Y Images", grid, self.epoch)
            
            grid = torchvision.utils.make_grid(torch.from_numpy(outputs[:8].to('cpu').detach().numpy().astype(np.float32)))
            self.logger.experiment.add_image("Predicted", grid, self.epoch)
            
            self.epoch += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(y))
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(torch.from_numpy(x[:8].to('cpu').numpy().astype(np.float32)))
            self.logger.experiment.add_image("Val X Images", grid, self.epoch)

            grid = torchvision.utils.make_grid(torch.from_numpy(y[:8].to('cpu').numpy().astype(np.float32)))
            self.logger.experiment.add_image("Val Y Images", grid, self.epoch)
            
            grid = torchvision.utils.make_grid(torch.from_numpy(outputs[:8].to('cpu').detach().numpy().astype(np.float32)))
            self.logger.experiment.add_image("Val Predicted", grid, self.epoch)
            
        return loss


    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        return preds
    
    def configure_optimizers(self):
        # return optim.SGD(self.parameters(), lr=self.lr, momentum=0.99)
        print(self.auto_lr)
        return optim.Adam(self.parameters(), lr=self.lr)
