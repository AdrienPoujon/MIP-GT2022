import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class Qu_Dataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 1]
        return image, label