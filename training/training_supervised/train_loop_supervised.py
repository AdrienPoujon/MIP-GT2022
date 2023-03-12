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
import numpy as np
from config.config import parse_option
from utils.utils import accuracy,AverageMeter,warmup_learning_rate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, labels) in enumerate(val_loader):
            images = image.float().to(device)

            labels = labels.long()

            label_list.append(labels.detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = model(images)

            loss = criterion(output, labels)
            _, pred = output.topk(1, 1, True, True)

            out_list.append(pred.detach().cpu().numpy())

            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    label_array = np.array(label_list)

    out_array = np.array(out_list)
    #prec = precision_score(label_array.flatten(), out_array.flatten())
    #rec = recall_score(label_array.flatten(), out_array.flatten())
    #print('Precision = '+ str(precision_score(label_array.flatten(),out_array.flatten())))
    #print('Recall = ' + str(recall_score(label_array.flatten(), out_array.flatten())))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def train(train_loader, model,criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = image.to(device)
        labels = labels.long()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss


        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        acc1= accuracy(output, labels, topk=(1,))


        top1.update(acc1[0].item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg