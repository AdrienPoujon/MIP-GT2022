from utils.utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys

def train_chexpert(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (images, sex, age, front, ap_pa, nothing, enlarged_cardiomedia, cardiomegaly, opacity, lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural_effusion, pleural_other, fracture, support_device) in enumerate(train_loader):
        data_time.update(time.time() - end)
        #print(images[0].shape)
        #plt.imshow(images[0][0].detach().squeeze().cpu().numpy())
        #plt.show()
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.to(device)
        bsz = opt.batch_size

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        #Method 1

        if opt.method1 == 'age':
            labels1 = age.cuda()
        elif opt.method1 == 'front_lat':
            labels1 = front.cuda()
        elif opt.method1 == 'ap_pa':
            labels1 = ap_pa.cuda()

        elif opt.method1 == 'nothing':
            labels1 = nothing.cuda()
        elif opt.method1 =='enlarged_cardiomedia':
            labels1 = enlarged_cardiomedia.cuda()
        elif opt.method1 =='cardiomegaly':
            labels1 = cardiomegaly.cuda()
        elif opt.method1 =='opacity':
            labels1 = opacity.cuda()
        elif opt.method1 =='lesion':
            labels1 = lesion.cuda()
        elif opt.method1 =='edema':
            labels1 = edema.cuda()
        elif opt.method1 =='consolidation':
            labels1 = consolidation.cuda()
        elif opt.method1 =='pneumonia':
            labels1 = pneumonia.cuda()
        elif opt.method1 =='atelectasis':
            labels1 = atelectasis.cuda()
        elif opt.method1 =='pneumothorax':
            labels1 = pneumothorax.cuda()
        elif opt.method1 =='pleural_effusion':
            labels1 = pleural_effusion.cuda()
        elif opt.method1 =='pleural_other':
            labels1 = pleural_other.cuda()
        elif opt.method1 =='fracture':
            labels1 = fracture.cuda()
        elif opt.method1 =='support_device':
            labels1 = support_device.cuda()
        else:
            labels1 = 'Null'
        # Method 2
        if opt.method2 == 'age':
            labels2 = age.cuda()
        elif opt.method2 == 'front_lat':
            labels2 = front.cuda()
        elif opt.method2 == 'ap_pa':
            labels2 = ap_pa.cuda()

        elif opt.method2 == 'nothing':
            labels2 = nothing.cuda()
        elif opt.method2 =='enlarged_cardiomedia':
            labels2 = enlarged_cardiomedia.cuda()
        elif opt.method2 =='cardiomegaly':
            labels2 = cardiomegaly.cuda()
        elif opt.method2 =='opacity':
            labels2 = opacity.cuda()
        elif opt.method2 =='lesion':
            labels2 = lesion.cuda()
        elif opt.method2 =='edema':
            labels2 = edema.cuda()
        elif opt.method2 =='consolidation':
            labels2 = consolidation.cuda()
        elif opt.method2 =='pneumonia':
            labels2 = pneumonia.cuda()
        elif opt.method2 =='atelectasis':
            labels2 = atelectasis.cuda()
        elif opt.method2 =='pneumothorax':
            labels2 = pneumothorax.cuda()
        elif opt.method2 =='pleural_effusion':
            labels2 = pleural_effusion.cuda()
        elif opt.method2 =='pleural_other':
            labels2 = pleural_other.cuda()
        elif opt.method2 =='fracture':
            labels2 = fracture.cuda()
        elif opt.method2 =='support_device':
            labels2 = support_device.cuda()
        else:
            labels2 = 'Null'
        if(opt.num_methods == 0):
            loss = criterion(features)
        elif(opt.num_methods==1):
            loss = criterion(features,labels1)
        elif(opt.num_methods == 2):
            loss = criterion(features,labels1) + criterion(features,labels2)
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