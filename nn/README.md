## Overview

This directory primarily contains code implementing the models in the paper, as well as the code to train or pre-train them. (In addition, this directory contains a python script for converting the text feature files produced by the code in the modifiedBCS/ directory into matrices in hdf5 format, to be consumed by the torch models. See text_feats_to_hdf5_README.md for further details on this python script).

The script ../run_experiments.py contains the commands necessary for duplicating our experiments from feature extraction through training and evaluation, and accordingly provides examples of how to use the code in this directory.

## Prerequisites
1. The code assumes you have access to the torch, nn, and hdf5 packages.
2. Most of the training code depends on the libcr module, which must be compiled before it can be used. To compile, type the following:
    ```
     luarocks make rocks/cr-scm-1.rockspec
    ```
3. Much of the code also assumes that the following 3 directories exist: models/, bps/, and conllouts/. These directories are used to store trained models when training, to store predictions in a temporary format, and to store predictions in CoNLL output format (so that the scorer script can be used), respectively.
4. The code assumes that you have the train, dev, and test features in hdf5 format, which you can accomplish by using text_feats_to_hdf5.py on the text feature files produced by ../modifiedBCS/ or by downloading the hdf5 features themselves. It also assumes that you have ``Oracle Predicted Cluster'' text files for train and dev (assumed by default to be TrainOPCs.txt and DevOPCs.txt), which are used as supervision.  
    
The latter 3 steps listed above are also implemented in ../run_experiments.py; see there for details.

## Important Files
- ana_model.lua contains code for pre-training on the anaphoricity subtask
- ante_model.lua contains code for pre-training on the antecedent-ranking subtask
- full_g1_model.lua contains code for training the full g1 model described in the paper, either using the pre-trained parameters or from a random initialization, as well as code for making predictions with the model
- full_g2_model.lua contains code for training the full g2 model described in the paper, either using the pre-trained paramters or from a random initialization, as well as code for making predictions with the model
    
All of the above files can be run with many options; see the documentation in the respective files.


