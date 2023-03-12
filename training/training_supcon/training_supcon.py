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
from models.resnet import SupCEResNet, SupConResNet
from config.config_supcon import parse_option
from utils.utils import accuracy,AverageMeter,adjust_learning_rate,save_model
import torch.nn as nn
import torch.optim as optim
import tqdm
from dataset.dataset import COVIDKaggleDataset
from dataset.Chexpert_dataset import ChexpertDataset
from dataset.chest_clusters import  Chexpert_Clusters_Dataset
from training.training_supcon.training_chexpert import train_chexpert
from training.training_supcon.training_one_epoch_chexpert_clusters import train_chexpert_Clusters
from loss.contrast_loss import SupConLoss

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'OCT' or opt.dataset == 'covid_kaggle' or opt.dataset == 'Chexpert' or opt.dataset == 'Chexpert_Clusters':
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
        train_dataset = COVIDKaggleDataset(opt.train_csv_path,img_dir,transform=TwoCropTransform(train_transform))
    elif opt.dataset == 'Chexpert':
        img_dir = '/data/Datasets/'
        train_dataset = ChexpertDataset(opt.train_csv_path, img_dir, TwoCropTransform(train_transform))
    elif opt.dataset == 'Chexpert_Clusters':
        img_dir = '/data/Datasets/'
        train_dataset = Chexpert_Clusters_Dataset(opt.train_csv_path, img_dir, TwoCropTransform(train_transform))
    else:
        train_dataset = None
        test_dataset = None
    train_sampler = None
    print(opt.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)


    return train_loader


def set_model(opt):
    criterion = SupConLoss(temperature=opt.temp, device=opt.device)
    model = SupConResNet(name=opt.model_type)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()
    return model, criterion

def set_optimizer(opt,model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
    #print(model)
    #summary(model,(1,128,128))
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        #loss = train_OCT(train_loader, model, criterion, optimizer, epoch, opt)
        if(opt.dataset =='Chexpert'):
            loss = train_chexpert(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()

