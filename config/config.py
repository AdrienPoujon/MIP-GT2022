
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--n_cls', type=int, default=4,
                        help='number of training epochs')
    parser.add_argument('--image_size', type=int, default=224,
                        help='number of training epochs')
    parser.add_argument('--lr_decay_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--super', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--type', type=int, default=0,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=.001,
                        help='learning rate')

    parser.add_argument('--ckpt', type=str, default='../../save/explainable/VGG_19_COVID-QU-Ex.tar',
                        help='learning rate')


    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay')
    parser.add_argument('--model_type', type=str, default='',
                        help='type of model')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--device', type=str, default='cuda:0')
    # model dataset

    # other setting
    parser.add_argument('--tb_folder', type=str, default='',
                        help='Tensorboard Logging')
    parser.add_argument('--dataset', type=str, default='COVID-QU-Ex',
                        help='Dataset Name')
    parser.add_argument('--test_path', type=str,default ='',
                        help='Test Image Folder Location')
    parser.add_argument('--train_path', type=str,default ='',
                        help='Train Image Folder Location')
    parser.add_argument('--train_csv_path', type=str, default='../../csv_files/COVID_KAGGLE/train.csv',
                        help='Test Image Folder Location')
    parser.add_argument('--val_csv_path', type=str, default='',
                        help='Test Image Folder Location')
    parser.add_argument('--test_csv_path', type=str, default='../../csv_files/COVID_KAGGLE/test.csv',
                        help='Train Image Folder Location')
    parser.add_argument('--test_path_labels', type=str,default ='',
                        help='Test Label Folder Location')
    parser.add_argument('--train_path_labels', type=str,default = '',
                        help='Train Label Folder Location')
    parser.add_argument('--save_path', type=str, default='./save/explainable/VGG_19_COVID-QU-Ex.tar',
                        help='SAVE Model Folder Location')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    opt = parser.parse_args()


    return opt