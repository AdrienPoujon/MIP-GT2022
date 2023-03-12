import os
import sys
import argparse
import time
import math
from torchsummary import summary
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from models.resnet import SupCEResNet
from config.config import parse_option
from utils.utils import accuracy,AverageMeter,adjust_learning_rate,save_model
import torch.nn as nn
import torch.optim as optim
import tqdm
from dataset.dataset import COVIDKaggleDataset
from train_loop_supervised import train, validate
def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'OCT' or opt.dataset == 'covid_kaggle':
        mean = (.1904)
        std = (.2088)
    else:
        mean = 0
        std = 1
        print('Wrong Dataset Specified')
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'covid_kaggle':
        img_dir = '/data/Datasets/COVID-19_Radiography_Dataset'
        train_dataset = COVIDKaggleDataset(opt.train_csv_path,img_dir,train_transform)
        test_dataset = COVIDKaggleDataset(opt.test_csv_path,img_dir,val_transform)
    else:
        train_dataset = None
        test_dataset = None
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, test_loader


def set_model(opt):
    model = SupCEResNet(name = opt.model_type,num_classes = opt.n_cls)
    criterion = nn.CrossEntropyLoss()
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    return model, criterion

def set_optimizer(opt,model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def main():
    opt = parse_option()
    train_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt,model)
    print(model)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss, acc = train(train_loader, model,criterion,optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
    validate(test_loader, model, criterion, opt)
    save_model(model, optimizer, opt, epoch, opt.save_path)


if __name__ == '__main__':
    main()