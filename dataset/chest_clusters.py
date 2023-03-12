import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class Chexpert_Clusters_Dataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        labels_5 = self.df.iloc[idx,19]
        labels_200 = self.df.iloc[idx, 20]
        labels_500 = self.df.iloc[idx, 21]
        labels_1000 = self.df.iloc[idx, 19]
        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 3]
        return image, labels_5,labels_200,labels_500,labels_1000

