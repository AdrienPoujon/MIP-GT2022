import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class COVID_QU_Ex_dataset(Dataset):
    def __init__(self, split, img_dir, transform=None):
        self.root = img_dir
        self.split = split
        all_files = []
        all_labels = []

        i = 0
        for folder in ['Normal', 'COVID-19', 'Non-COVID']:  # 0 = normal, 1 = covid, 2 = non-covid
            files = os.listdir(os.path.join(self.root, self.split, folder, 'images'))
            labels = [i]*len(files)
            i += 1
            all_files.extend(files)
            all_labels.extend(labels)

        self.X = all_files
        self.Y = all_labels
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        img_name = x
        if x.startswith('covid'):
            folder = 'COVID-19'
        elif x.startswith('non_') or x.startswith('Non_'):
            folder = 'Non-COVID'
        else:
            folder = 'Normal'
        x = np.array(Image.open(os.path.join(self.root, self.split, folder, 'images', x)))

        if self.transform is not None:
            im_x = Image.fromarray(x)
            x = self.transform(im_x)

        return x, y,  os.path.join(self.root, self.split, folder, 'lung masks', img_name)

    def __len__(self):
        return len(self.X)

