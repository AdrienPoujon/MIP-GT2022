import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
import logging
import torch
from torchvision import models, transforms
import time
import os
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
from models.gradient_accessible_models import VGG
from dataset.dataset import COVIDKaggleDataset
import pandas as pd
from config.config import parse_option
from visualization.gradcam_contrast import gen_heatmap_contrast, gen_heatmap_gradcam
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
import numpy as np
from dataset.COVID_QU_Ex_dataset import COVID_QU_Ex_dataset
import cv2
import matplotlib.pyplot as plt
from PIL import Image
logger = logging.getLogger(__name__)


def main():
    opt=parse_option()
    model_save_dir = opt.ckpt
    checkpoint = torch.load(model_save_dir, map_location='cuda:0')
    model = models.vgg19(pretrained=True)
    model = VGG(model)

    if (opt.dataset == 'covid_kaggle'):
        model.classifier[6] = nn.Linear(4096, 4)
        model.features_conv[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1))
    elif (opt.dataset == 'COVID-QU-Ex'):
        model.classifier[6] = nn.Linear(4096, 3)
        model.features_conv[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1))

    model.load_state_dict(checkpoint)


    model.eval()
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.1706], std=[.2112])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.1706], std=[.2112])
        ]),
    }
    test_df = opt.test_csv_path
    if (opt.dataset == 'covid_kaggle'):
        img_dir = '/home/admin/Downloads/data/Datasets/COVID-19_Radiography_Dataset'
        dataset_val = COVIDKaggleDataset(test_df, img_dir, data_transforms['val'])
    elif (opt.dataset == 'COVID-QU-Ex'):
        img_dir = '../../dataset/COVID-QU-Ex/Infection Segmentation Data'
        dataset_val = COVID_QU_Ex_dataset('Test', img_dir, data_transforms['val'])


    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                             num_workers=0,
                                             drop_last=True)
    device = 'cuda:0'
    model = model.to(device)
    for i, (inputs, label, img_name) in enumerate(tqdm(val_loader)):
        img = inputs.to(device)

        label = label.to(device)
        pred = model(img)
        val = pred.argmax(dim=1)
        if(label == val and label == 0):
            map = gen_heatmap_gradcam(model, img, img_name[0])
            # con_map = gen_heatmap_contrast(model, img, label, 2, device, img_name[0])
            img = img.squeeze().detach().cpu().numpy()
            img = np.stack((img,)*3, axis=-1)
            # img = np.moveaxis(img, 0, 2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # plt.figure(1)
            # plt.imshow(img, cmap='gray')
            # plt.imshow(con_map, cmap='jet', alpha=.3)
            # plt.title("Why Covid Rather than Non-Covid? " + img_name[0].split('/')[-1])
            plt.figure(2)
            plt.imshow(img, cmap='gray')
            plt.imshow(map, cmap='jet', alpha=.3)
            plt.title("Why Normal? " + img_name[0].split('/')[-1])
            plt.show()
            # plt.figure(3)
            # tmp = img_name[0].split('/')
            # tmp[-2] = 'infection masks'
            # img_name = '/'.join(tmp)
            # plt.imshow(np.array(Image.open(img_name)), cmap='gray')
            # plt.title("Infection Mask " + img_name.split('/')[-1])
            # plt.show()


if __name__ == '__main__':
    main()