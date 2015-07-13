## text_feats_to_hdf5.py

This script runs many parallel jobs in order to create hdf5 vectors containing the features in the text files output by the modifiedBCS system.  These hdf5 vectors are used by the neural networks in the nn/ directory.

In addition to providing the name of the text feature file and a prefix for identifying the feature-set, the script needs to know whether it is converting pairwise or anaphoricity features and whether it is converting a training file or not. 

Below we provide the commands necessary to generate the necessary hdf5 vectors for the Basic+ features output by modifiedBCS, as described in the README in the modifiedBCS/ directory. Further options and documentation can be found in the script itself.

```
# use 30 processes to generate pairwise Basic+ train features
python text_feats_to_hdf5.py -n 30 NONE-FINAL,MOARANAPH,MOARPW+bilexical-pwTrainFeats.txt train_basicp pw
```

```
# use 30 processes to generate pairwise Basic+ dev features
python text_feats_to_hdf5.py -n 30 -t NONE-FINAL,MOARANAPH,MOARPW+bilexical-pwDevFeats.txt dev_basicp pw
```

```
# use 30 processes to generate pairwise Basic+ test features
python text_feats_to_hdf5.py -n 30 -t NONE-FINAL,MOARANAPH,MOARPW+bilexical-pwTestFeats.txt test_basicp pw
```

```
# use 30 processes to generate anaphoricity Basic+ train features
python text_feats_to_hdf5.py -n 30 NONE-FINAL,MOARANAPH,MOARPW+bilexical-anaphTrainFeats.txt train_basicp ana
```

```
# use 30 processes to generate pairwise Basic+ dev features
python text_feats_to_hdf5.py -n 30 -t NONE-FINAL,MOARANAPH,MOARPW+bilexical-anaphDevFeats.txt dev_basicp ana
```

```
# use 30 processes to generate pairwise Basic+ test features
python text_feats_to_hdf5.py -n 30 -t NONE-FINAL,MOARANAPH,MOARPW+bilexical-anaphTestFeats.txt test_basicp ana
```

