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

        img_path = os.path.join(self.img_dir,self.df.iloc[idx,0])
        age = self.df.iloc[idx,1]
        front_lat = self.df.iloc[idx,2]
        ap_pa = self.df.iloc[idx,3]
        nothing = self.df.iloc[idx,4]
        enlarged_cardiomedia = self.df.iloc[idx,5]
        opacity = self.df.iloc[idx,6]
        lesion = self.df.iloc[idx,7]
        edema = self.df.iloc[idx,8]
        consolidation = self.df.iloc[idx,9]
        pneumonia = self.df.iloc[idx,10]
        atelectasis = self.df.iloc[idx,11]
        pneumothorax = self.df.iloc[idx,12]
        pleural_effusion = self.df.iloc[idx,13]
        pleural_other = self.df.iloc[idx,14]
        fracture = self.df.iloc[idx,15]
        support_device = self.df.iloc[idx,16]



        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 3]
        return image, age, front_lat,ap_pa,nothing,enlarged_cardiomedia,opacity,lesion,edema,consolidation,pneumonia,atelectasis,pneumothorax,pleural_effusion,pleural_other,fracture,support_device