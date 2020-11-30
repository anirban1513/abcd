from __future__ import print_function, division
import numpy as np
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def normalize(img, minval=-100, maxval=240, bit_conv=True):
    if bit_conv:
        img = (np.uint8(img)/255.).astype(np.float)
    else:
        img = (img - minval)/(maxval - minval)
    return img

class LungDataset(Dataset):
    def __init__(self, df, slices=3, transform=None):
        self.df = df
        self.label_list = self.df['label'].tolist()
        self.classes = self.df['class'].tolist()
        self.data_list = self.df['img'].tolist()
        self.slices = slices
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        self.data = np.load(self.data_list[item])['img']
        self.out = np.load(self.label_list[item])['label'].astype(np.float)
        
        self.out = self.out.reshape(*self.out.shape, 1).astype(np.float)
        self.data = normalize(self.data, bit_conv=False).reshape(320, 320, self.slices)

        if self.transform:
            augmented = self.transform(image=self.data, mask=self.out)
            self.data = augmented['image']
            self.out = augmented['mask']
        return self.data_list[item], self.data.reshape(512, 512, self.slices).transpose(2, 0, 1), self.out.reshape(1, 512, 512)
    
    def get_labels(self):
        return [int(i) for i in self.classes]
