import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class ChexpertDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        sex = self.df.iloc[idx, 1]
        age = self.df.iloc[idx, 2]
        front = self.df.iloc[idx,3]

        ap_pa = self.df.iloc[idx, 4]
        nothing = self.df.iloc[idx, 5]
        enlarged_cardiomedia = self.df.iloc[idx, 6]
        cardiomegaly = self.df.iloc[idx,7]
        opacity = self.df.iloc[idx, 8]
        lesion = self.df.iloc[idx, 9]
        edema = self.df.iloc[idx, 10]
        consolidation = self.df.iloc[idx, 11]
        pneumonia = self.df.iloc[idx, 12]
        atelectasis = self.df.iloc[idx, 13]
        pneumothorax = self.df.iloc[idx, 14]
        pleural_effusion = self.df.iloc[idx, 15]
        pleural_other = self.df.iloc[idx, 16]
        fracture = self.df.iloc[idx, 17]
        support_device = self.df.iloc[idx, 18]

        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 3]
        return image, sex, age, front, ap_pa, nothing, enlarged_cardiomedia, cardiomegaly, opacity, lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural_effusion, pleural_other, fracture, support_device


class COVIDKaggleDataset(data.Dataset):
    def __init__(self, df, img_dir, transform):

        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transform

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