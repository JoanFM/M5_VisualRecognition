# M5_VisualRecognition -- Group 7
## Project Distribution
- [x] Week1: Reproduce the M3 Final Model
- [x] Week2: Try Detectron2

## Members:
- [Roger Casals](rogercasalsvilardell@gmail.com): rogercasalsvilardell@gmail.com
- [Joan Fontanals](jfontanalsmartinez@gmail.com): jfontanalsmartinez@gmail.com
- [Pau Domingo](pavdom72@gmail.com): pavdom72@gmail.com

## Instructions:
Under the folders WeekX we have the work of each week.

### Week 1
- The structures of the models are located in folder 'models'. The main file is used to train the network. 
- Google slides presenting the work done can be seen in the following link: https://docs.google.com/presentation/d/1BTrpLU-RDcfGMtZMdrTy72b93fPcUCndJGkbUnBvUX0/edit?usp=sharing

### Week 2
- Google slides presenting the work done can be seen in the global google slides for the project, as well as more detailed in the following link: https://docs.google.com/presentation/d/1FxOvYU1YnDSaeh2kJF-MFt_KoKiX2hdXRXg9ZDpw4VA/edit?usp=sharing

### Week 3
- Google slides presenting the work done can be seen in the global slides for the project, as well as more detailed slides in the following link: https://docs.google.com/presentation/d/1YHKYtZGwY71cXIbK4Sxv5A9ltE_tuOHZMETPXQ9ftdo/edit#slide=id.g70d18d995b_0_1

### Week 4
- Google slides presenting the work done can be seen in the global slides for the project, as well as more detailed slides in the following link: https://docs.google.com/presentation/d/1ZUMDiKfwdLk92_UiYmmyslSHn0dS-actZeCsOI_Fj9A/edit#slide=id.g71aeadc0fc_0_109

### Week 5
- Google slides presenting the work done can be seen in the global slides for the project, as well as more detailed slides in the following link: https://docs.google.com/presentation/d/1bxxUZpxozHgp5d3HXVT6fNFPuT1I-fMn7TkjpsytwTY/edit#slide=id.g72385ef41b_0_80

### Overleaf
Overleaf link to view project report: https://www.overleaf.com/read/cdsfkrqrngrp

## Run tensorboard:
Indcluded a file experiment containing an example of logs and events tracked by tensorboard. To look at them in your browser run:
tensorboard --logdir experiment/ --port XXXX

## SetUp DeepLab v3+:
To run training with DeepLabV3+ and to reproduce the results obtained in the paper [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf) with CityScapes dataset one needs to follow these steps:

1 - Clone tensorflow models from https://github.com/tensorflow/models.git.

2 - From tensorflow/models/research/deeplab/datasets:

  2.1 - Obtain or download cityscapes dataset from https://www.cityscapes-dataset.com/ (Note that you need to register)
  
  2.2 - Clone https://github.com/mcordts/cityscapesScripts.git
  
  2.3 - Run sh convert_cityscapes.sh
  
  2.4 - This will generate a folder tfrecords with a recommended structure like this:
  
        ```
        + datasets
          + cityscapes
            + leftImg8bit
              + gtFine
              + tfrecord
              + exp
              + train_on_train_set
                + train
                + eval
                + vis
        ```
        
  2.5 - Inside the tfrecord folder move train* to train_fine* and val* to val_fine* to be able to train on train_fine dataset.
  
  2.6 - Do not forhet to export PYTHONPATH as ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` from tensorflow/models/research
