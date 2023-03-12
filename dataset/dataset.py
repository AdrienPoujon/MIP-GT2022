import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch

class COVIDKaggleDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):

        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if(self.df.iloc[idx,2] == 'Normal'):
            split = self.df.iloc[idx,0].split('-')
            file_new = 'Normal' + '-' + split[1]
            file = file_new  + '.png'
        else:
            file = self.df.iloc[idx,0] + '.png'
        img_path = os.path.join(self.img_dir,self.df.iloc[idx,2],file)
        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 3]
        return image, label


