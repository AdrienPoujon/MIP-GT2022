import math
from torchsummary import summary
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from config.config_linear import parse_option
from utils.utils import accuracy,AverageMeter,warmup_learning_rate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from models.resnet import SupCEResNet
import time
import sys
from utils.utils_updated import set_loader_new, set_model, set_optimizer, adjust_learning_rate
from utils.utils_updated import AverageMeter,warmup_learning_rate,accuracy
def main_chest_super():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    train_loader,  test_loader = set_loader_new(opt)


    # build model and criterion
    #model, classifier, criterion = set_model(opt)

    # build optimizer
    #optimizer = set_optimizer(opt, classifier)
    acc_list = []
    prec_list = []
    rec_list = []
    # training routine
    for i in range(0,1):
        model = SupCEResNet(name='resnet18', num_classes=4)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

    # eval for one epoch
        #loss, val_acc = validate(test_loader, model, classifier, criterion, opt)
    #if val_acc > best_acc:
        #best_acc = val_acc
        #best_class = classifier
        loss, test_acc = validate(test_loader, model, criterion, opt)



    with open("/home/kiran/Desktop/Dev/SupCon/results.txt", "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write('Accuracy: ' + str(test_acc) + '\n')
        file.write('Training Accuracy: ' + str(acc) + '\n' )

    print('Accuracy: ' + str(test_acc))
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()
    y_pred = []
    y_true = []
    for idx, (image, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = labels.long()
        bsz = labels.shape[0]
        labels=labels.to(device)
        y_true.append(labels[0].detach().cpu().numpy())
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        #with torch.no_grad():
        output = model(images)
        _, pred = output.topk(1, 1, True, True)

        y_pred.append(pred[0].detach().cpu().numpy())
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
    y_pred_0 = [1 if x == 0 else 0 for x in y_pred]
    y_true_0 = [1 if x == 0 else 0 for x in y_true]
    # Label 1
    y_pred_1 = [1 if x == 1 else 0 for x in y_pred]
    y_true_1 = [1 if x == 1 else 0 for x in y_true]
    # Label 2
    y_pred_2 = [1 if x == 2 else 0 for x in y_pred]
    y_true_2 = [1 if x == 2 else 0 for x in y_true]
    # Label 3
    y_pred_3 = [1 if x == 3 else 0 for x in y_pred]
    y_true_3 = [1 if x == 3 else 0 for x in y_true]

    with open("/home/kiran/Desktop/Dev/SupCon/results.txt", "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write(opt.dataset + '\n')
        file.write(opt.train_csv_path + '\n')
        file.write('Training Results')
        file.write('Accuracy: ' + str(top1.avg) + '\n')
        # file.write('Averaged Precision: ' + str(precision_score(y_true, y_pred, average='macro')))
        # file.write('Averaged Recall: ' + str(recall_score(y_true, y_pred, average='macro')))
        # file.write('Averaged F1-Score: ' + str(f1_score(y_true, y_pred, average='macro')))
        # file.write('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
        # Precision, Recall, F1-Score Per Class
        file.write('0 Precision: ' + str(precision_score(y_true_0, y_pred_0)))
        file.write('0 Recall: ' + str(recall_score(y_true_0, y_pred_0)))
        file.write('0 F1-Score: ' + str(f1_score(y_true_0, y_pred_0)))
        file.write('\n')

        file.write('1 Precision: ' + str(precision_score(y_true_1, y_pred_1)))
        file.write('1 Recall: ' + str(recall_score(y_true_1, y_pred_1)))
        file.write('1 F1-Score: ' + str(f1_score(y_true_1, y_pred_1)))
        file.write('\n')
        file.write('2 Precision: ' + str(precision_score(y_true_2, y_pred_2)))
        file.write('2 Recall: ' + str(recall_score(y_true_2, y_pred_2)))
        file.write('2 F1-Score: ' + str(f1_score(y_true_2, y_pred_2)))
        file.write('\n')
        file.write('3 Precision: ' + str(precision_score(y_true_3, y_pred_3)))
        file.write('3 Recall: ' + str(recall_score(y_true_3, y_pred_3)))
        file.write('3 F1-Score: ' + str(f1_score(y_true_3, y_pred_3)))
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, labels) in enumerate(val_loader):
            images = image.float().to(device)


            labels = labels.long()

            y_true.append(labels[0].detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = model(images)

            loss = criterion(output, labels)
            _, pred = output.topk(1, 1, True, True)

            y_pred.append(pred[0].detach().cpu().numpy())
            # update metri  c
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
        # Label 0
        y_pred_0 = [1 if x == 0 else 0 for x in y_pred]
        y_true_0 = [1 if x == 0 else 0 for x in y_true]
        # Label 1
        y_pred_1 = [1 if x == 1 else 0 for x in y_pred]
        y_true_1 = [1 if x == 1 else 0 for x in y_true]
        # Label 2
        y_pred_2 = [1 if x == 2 else 0 for x in y_pred]
        y_true_2 = [1 if x == 2 else 0 for x in y_true]
        # Label 3
        y_pred_3 = [1 if x == 3 else 0 for x in y_pred]
        y_true_3 = [1 if x == 3 else 0 for x in y_true]
        with open("/home/kiran/Desktop/Dev/SupCon/results.txt", "a") as file:
            # Writing data to a file
            file.write(opt.ckpt + '\n')
            file.write(opt.biomarker + '\n')
            file.write(opt.dataset + '\n')
            file.write(opt.train_csv_path + '\n')
            file.write('Accuracy: ' + str(top1.avg) + '\n')
            # file.write('Averaged Precision: ' + str(precision_score(y_true, y_pred, average='macro')))
            # file.write('Averaged Recall: ' + str(recall_score(y_true, y_pred, average='macro')))
            # file.write('Averaged F1-Score: ' + str(f1_score(y_true, y_pred, average='macro')))
            # file.write('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
            # Precision, Recall, F1-Score Per Class
            file.write('0 Precision: ' + str(precision_score(y_true_0, y_pred_0)))
            file.write('0 Recall: ' + str(recall_score(y_true_0, y_pred_0)))
            file.write('0 F1-Score: ' + str(f1_score(y_true_0, y_pred_0)))
            file.write('\n')

            file.write('1 Precision: ' + str(precision_score(y_true_1, y_pred_1)))
            file.write('1 Recall: ' + str(recall_score(y_true_1, y_pred_1)))
            file.write('1 F1-Score: ' + str(f1_score(y_true_1, y_pred_1)))
            file.write('\n')
            file.write('2 Precision: ' + str(precision_score(y_true_2, y_pred_2)))
            file.write('2 Recall: ' + str(recall_score(y_true_2, y_pred_2)))
            file.write('2 F1-Score: ' + str(f1_score(y_true_2, y_pred_2)))
            file.write('\n')
            file.write('3 Precision: ' + str(precision_score(y_true_3, y_pred_3)))
            file.write('3 Recall: ' + str(recall_score(y_true_3, y_pred_3)))
            file.write('3 F1-Score: ' + str(f1_score(y_true_3, y_pred_3)))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = ('Normal', 'Covid-19', 'Non-Covid', 'Viral')
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.imshow(df_cm)
    # plt.show()
    #file = str(opt.ckpt[0:50]) + '.png'
    #file = opt.ckpt.split('/')
    #file = file[9] + '.png'
    #print(file)
    plt.savefig('super.png')
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg