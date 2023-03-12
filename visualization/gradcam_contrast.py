import torch.nn as nn
import logging
import torch
from torchvision import models, transforms
import numpy as np
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

def gen_heatmap_gradcam(model,img, img_name=None):
    pred = model(img)

    val = pred.argmax(dim=1)

    pred[:, val].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(img).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    # Heat map stuff

    mask = np.array(Image.open(img_name))
    mask[mask == 255] = 1

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.detach().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.numpy()
    img = img.squeeze().detach().cpu().numpy()
    img = np.stack((img,)*3, axis=-1)
    # img = np.moveaxis(img, 0, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    heatmap = heatmap * mask

    heatmap = np.uint8(255 * heatmap)
    return heatmap


def gen_heatmap_contrast(model, img, label, contrast, device, img_name=None):
    contrast_tensor = torch.from_numpy(np.asarray([contrast])).to(device)
    model = model.to(device)
    pred = model(img)

    val = pred.argmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    im_label_as_var = Variable(contrast_tensor)
    tensor = torch.tensor(im_label_as_var, dtype=torch.long, device=device)
    pred_loss = criterion(pred, tensor)
    model.zero_grad()
    pred_loss.backward()

    gradients = model.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(img).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    mask = np.array(Image.open(img_name))
    mask[mask == 255] = 1


    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.detach().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.numpy()
    img = img.squeeze().detach().cpu().numpy()
    img = np.stack((img,)*3, axis=-1)
    # img = np.moveaxis(img, 0, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    heatmap = heatmap * mask

    heatmap = np.uint8(255 * heatmap)
    return heatmap