# ECE6780_MedEmbeddings01
ECE 6780 Group 1

Go to starting directory and type:

export PYTHONPATH=$PYTHONPATH:$PWD

before doing any experiments.

Create a directory called "model_checkpoints" inside of your starting directory.

bash run_script

Do this to run whatever codes you wish to.

Datasets:

Chexpert: https://stanfordmlgroup.github.io/competitions/chexpert/ Covid Kaggle: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database Covid Qu Dataset: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu Covid X Dataset: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2?select=competition_test

Config:

Study config and config_linear files to set the parameters that are of interest. Noteworthy features are it's necessary to specify the training and testing csv files, checkpoints for the saved models, and other hyperparamters such as the batch size, learning rate, and others.

Supervised Contrastive Pre-training:

Study the files in training/training_supcon.

training_supcon runs the main code and depending on the choice of Chexpert or Chexpert_Clusters as the dataset, the resultant choice of the methods to run as part of the supervised contrastive loss is changed.

To specify SimCLR as the method of choice set the num_methods parameter to 0. This will involve supervised contrastive learning without any labels which is essentially the SimCLR framework.

Linear Fine-Tuning:

Study the files in training/training_linear and training/training_main

main_linear.py runs the code and loads the pre-trained model.

The files training_linear changes based on the dataset and whether we are using the supervised contrastive model or training from scratch with a fully supevised model.

Explainability Experiments:

The code in training/training_explainable is used to generate GradCAM and contrastive explanation heatmaps. The actual implementation of this is located in visualization/ContrastiveCAM and visualization/GradCAM.

The additional visualizations regarding clustering analysis are located as .ipynb files in the visualization folder.

Utilities:

The utils.py and utils_updated.py folders hold code for loading the models and the dataloaders as well as useful functions that manage the different options that can be chosen from the config file.

Run_Script:

All functions can be run from run_script. See the examples to get an idea how to run the code. Studying each of these sections in depth will identify the necessary flags to run each code.

Moco v2:

Code for running the moco v2 experiments are located in the directory.