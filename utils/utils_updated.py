from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import tqdm
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from models.resnet import  SupConResNet,LinearClassifier,LinearClassifier_MultiLabel
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from dataset.chest import ChexpertDataset,COVIDKaggleDataset
from dataset.chest_qu import Qu_Dataset
from dataset.covid_x import Covid_X_Dataset
import torch.nn as nn
def set_model(opt):
    if(opt.dataset == 'covid_kaggle'):
        model = SupConResNet(name=opt.model)
        criterion = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=4)
    elif(opt.dataset == 'qu_dataset'):
        model = SupConResNet(name=opt.model)
        criterion = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=3)
    elif (opt.dataset == 'covid_x'):
        model = SupConResNet(name=opt.model)
        criterion = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=2)
    elif(opt.dataset == 'covid_x_A'):
        model = SupConResNet(name=opt.model)
        criterion = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=3)
    else:
        if(opt.multi == 0 and opt.dataset !='Ford' and opt.dataset !='Ford_Region'):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.CrossEntropyLoss()

            classifier = LinearClassifier(name=opt.model, num_classes=2)
        elif(opt.dataset == 'Ford' or opt.dataset == 'Ford_Region'):
            print('Hello')
            model = SupConResNet(name=opt.model)
            model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            criterion = torch.nn.CrossEntropyLoss()
            if(opt.ford_region == 1):
                classifier = LinearClassifier(name=opt.model, num_classes=6)
            else:
                classifier = LinearClassifier(name=opt.model, num_classes=3)
        elif(opt.multi == 1 and opt.super == 3):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.BCELoss()
            classifier = LinearClassifier(name=opt.model, num_classes=1)
        elif(opt.multi == 1):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.BCELoss()

            classifier = LinearClassifier_MultiLabel(name=opt.model, num_classes=5)
    print(opt.ckpt)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(device)
        classifier = classifier.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion

def set_loader_new(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford' or opt.dataset == 'Ford_Region':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'Chexpert' or opt.dataset == 'covid_kaggle' or opt.dataset == 'qu_dataset' or opt.dataset =='covid_x' or opt.dataset == 'covid_x_A':
        mean = (.5093)
        std = (.2534)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.Resize(size=(224,224)),
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
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)

    elif opt.dataset == 'covid_kaggle':
        img_dir = opt.img_dir #'/data/Datasets/COVID-19_Radiography_Dataset'
        train_csv_path = opt.train_csv_path #'/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/train.csv'
        test_csv_path =  opt.test_csv_path #'/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/test.csv'
        train_dataset = COVIDKaggleDataset(train_csv_path, img_dir, train_transform)
        test_dataset = COVIDKaggleDataset(test_csv_path, img_dir, val_transform)
    elif opt.dataset == 'covid_x':
        img_dir_train = os.path.join(opt.img_dir,'train')
        img_dir_test = os.path.join(opt.img_dir,'test')
        train_csv_path = opt.train_csv_path
        test_csv_path =  opt.test_csv_path
        train_dataset = Covid_X_Dataset(train_csv_path, img_dir_train, train_transform)
        test_dataset = Covid_X_Dataset(test_csv_path, img_dir_test, val_transform)
    elif opt.dataset == 'covid_x_A':
        img_dir_train = os.path.join(opt.img_dir,'train')
        img_dir_test = os.path.join(opt.img_dir,'test')
        train_csv_path = opt.train_csv_path
        test_csv_path =  opt.test_csv_path
        train_dataset = Covid_X_Dataset(train_csv_path, img_dir_train, train_transform)
        test_dataset = Covid_X_Dataset(test_csv_path, img_dir_test, val_transform)
    elif opt.dataset == 'qu_dataset':
        img_dir = opt.img_dir
        train_csv_path = opt.train_csv_path
        test_csv_path = opt.test_csv_path
        train_dataset = Qu_Dataset(train_csv_path, img_dir, train_transform)
        test_dataset = Qu_Dataset(test_csv_path, img_dir, val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    if(opt.biomarker == 'drt' and opt.patient_split == 1):
        dl = True
    elif(opt.multi == 1):
        dl = True
    else:
        dl=False
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True,drop_last=dl)

    return train_loader, test_loader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def accuracy_multilabel(output,target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    r = roc_auc_score(target,output,multi_class='ovr')
    print(r)