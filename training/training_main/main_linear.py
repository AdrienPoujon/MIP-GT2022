from __future__ import print_function
from config.config_linear import parse_option

from training.training_linear.training_one_epoch_chest_linear import main_chest
from training.training_linear.training_one_epoch_chest_linaer_supervised import main_chest_super
from training.training_linear.training_one_epoch_covid_x import main_covid_x
from training.training_linear.training_one_epoch_covid_x_supervised import main_covid_x_super
from training.training_linear.training_one_epoch_qu import main_qu
from training.training_linear.training_one_epoch_qu_supervised import main_qu_super
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass







if __name__ == '__main__':
    opt = parse_option()
    if(opt.dataset == 'qu_dataset' and opt.super == 0):
        main_qu()
    elif(opt.dataset == 'qu_dataset' and opt.super == 1):
        main_qu_super()
    if(opt.dataset == 'covid_x' or opt.dataset == 'covid_x_A' and opt.super == 0):
        main_covid_x()
    elif(opt.dataset == 'covid_x' or opt.dataset == 'covid_x_A' and opt.super ==1 ):
        main_covid_x_super()
    if(opt.dataset == 'covid_kaggle' and opt.super == 0):
        main_chest()
    elif(opt.dataset == 'covid_kaggle' and opt.super == 1):
        main_chest_super()
