import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorboardX import SummaryWriter
# import pdb
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
from config.config import parse_option
from dataset.COVID_QU_Ex_dataset import COVID_QU_Ex_dataset
# from dataset.dataset import COVIDKaggleDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parse_option()
    ####################################################################################################################
    # Define Setup for Model
    ####################################################################################################################
    model = models.vgg19(pretrained=True)
    model = VGG(model)
    if(opt.dataset == 'covid_kaggle'):
        model.classifier[6] = nn.Linear(4096, 4)
        model.features_conv[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif (opt.dataset == 'COVID-QU-Ex'):
        model.classifier[6] = nn.Linear(4096, 3)
        model.features_conv[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model = model.to(device)
    print(model)
    ####################################################################################################################
    # Load Data
    ####################################################################################################################
    # if(opt.dataset == 'covid_kaggle'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.1706],std=[.2112])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.1706],std=[.2112])
        ]),
    }

    if(opt.dataset == 'covid_kaggle'):
        img_dir = '/home/admin/Downloads/data/Datasets/COVID-19_Radiography_Dataset'
        dataset_train = COVIDKaggleDataset(opt.train_csv_path, img_dir, data_transforms['train'])
        dataset_val = COVIDKaggleDataset(opt.test_csv_path, img_dir, data_transforms['val'])
    elif (opt.dataset == 'COVID-QU-Ex'):
        img_dir = './dataset/COVID-QU-Ex/Infection Segmentation Data'  #'../../dataset/COVID-QU-Ex/Infection Segmentation Data'
        dataset_train = COVID_QU_Ex_dataset('Train', img_dir, data_transforms['train'])
        dataset_val = COVID_QU_Ex_dataset('Val', img_dir, data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=opt.num_workers, drop_last=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}

    ####################################################################################################################
    # Set Training Hyperparameters
    ####################################################################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)
    writer = SummaryWriter(log_dir='../../logs')
    new_model = train_model(model, criterion, optimizer_ft, dataset_sizes, dataloaders, device, exp_lr_scheduler,
                            opt.epochs,
                            writer)
    output_checkpoint = opt.save_path
    # if not os.path.exists('../../save/explainable'):
    #     os.mkdir('../../save/explainable')
    if not os.path.exists('./save/explainable'):
        os.mkdir('./save/explainable')
    torch.save(new_model.state_dict(), output_checkpoint)


def train_model(model, criterion, optimizer,dataset_sizes, dataloaders, device, scheduler, num_epochs,writer):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    count_val_decrease = 0
    val_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # pdb.set_trace()

            # for inputs, label,region_class,region,distance in tqdm(dataloaders[phase]):
            for inputs, labels in tqdm(dataloaders[phase]):
            # for inputs, label in dataloaders[phase]:
                inputs = inputs.to(device)
                inputs = inputs.float()
                # labels = region_class.to(device)
                labels = labels.to(device)
                labels = labels.long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            if phase == 'val':
                val_losses.append(running_loss)
                scheduler.step()
                '''
                if(epoch>2):
                    if(val_losses[epoch]>val_losses[epoch-1]):
                        count_val_decrease = count_val_decrease + 1
                    else:
                        count_val_decrease = 0
                if(count_val_decrease == 6):
                    scheduler.step()
                    count_val_decrease = 0
                '''
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase =='train':
                writer.add_scalar('train/%s' % 'Training_Loss', epoch_loss, epoch)
            else:
                writer.add_scalar('val/%s' % 'Validation_Loss', epoch_loss, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    main()