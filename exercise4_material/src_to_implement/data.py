from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description

    def __init__(self, data: pd.DataFrame, mode: str):
        super().__init__()

        self.data = data
        self.mode = mode
        self._transfrom = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean,train_std),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = self.data.iloc[index]

        img = imread(data["filename"], as_gray = True)
        img = gray2rgb(img)
        img = self._transfrom(img)
        label = torch.tensor([data["crack"],data["inactive"]])

        return img, label