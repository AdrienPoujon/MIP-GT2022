from utils.utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys
def train_chexpert_Clusters(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (images, labels_5,labels_200,labels_500,labels_1000) in enumerate(train_loader):
        data_time.update(time.time() - end)
        #print(images[0].shape)
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.to(device)

        bsz = labels_5.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        #Method 1

        if opt.method1 == '5':
            labels1 = labels_5.cuda()
        elif opt.method1 == '200':
            labels1 = labels_200.cuda()
        elif opt.method1 == '500':
            labels1 = labels_500.cuda()
        elif(opt.method1 == '1000'):
            labels1 = labels_1000.cuda()
        else:
            labels1 = 'None'



        if(opt.num_methods == 0):
            loss = criterion(features)
        elif(opt.num_methods==1):

            loss = criterion(features,labels1)
        else:
            loss = 'Null'
        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg