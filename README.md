# nn_coref
Neural Coref Models, as described in 
["Learning Global Features for Coreference Resolution"](http://nlp.seas.harvard.edu/papers/corefmain.pdf), Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber, NAACL 2016,

and

["Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution"](http://people.seas.harvard.edu/~srush/acl15.pdf), Sam Wiseman, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. ACL 2015.

For questions/concerns/bugs please contact swiseman at seas.harvard.edu.


## Overview
To keep things simple, the original ACL code is now in the acl15 branch. This README will cover duplicating the NAACL 2016 results.

## Generating Features
See the README in the modifiedBCS/ directory for running the Scala feature/mention extractor. Once you've generated text feature files, use text_feats_to_hdf_5_replacezero.py to convert them to hdf5 (to be consumed by Torch), as follows:

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt train_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt dev_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphTestFeats.txt test_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwTrainFeats.txt train_small pw -n 4 -r 28394```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwDevFeats.txt dev_small pw -n 4 -r 28394```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwTestFeats.txt test_small pw -n 4 -r 28394```

The "-r" argument takes the index of a dummy feature used to replace features unseen in the training set; above it is set to be one greater than the number of training features (and should never be less than this). The "-n" argument controls the number of processes spawned by the script.

You can also download bzipped hdf5 features here: https://drive.google.com/folderview?id=0B1ytQXPDuw7OVzI3MlRLMEFCcHM&usp=sharing 

**Before doing any training or pre-training, please create a directory called nn/models/**

## Pre-training
Given the hdf5 files generated in the previous step, you can pre-train anaphoricity and pairwise networks as follows:

```th ana_model.lua```

```th ante_model.lua -gpuid 0```

See the respective files for additional options and documentation.

You can download bzipped pre-trained anaphoricity and pairwise networks from https://drive.google.com/folderview?id=0B1ytQXPDuw7OYUcwSEVPRjFEM00&usp=sharing , where they are called small_200.model-na-0.100000.bz2 and small_700.model-pw-0.100000.bz2, respectively.

## Training the Full Model
Assuming you've put your pre-trained networks in nn/models/, you can now train the full model as follows:

```th mr_clust_embed.lua -gpuid 0 -PT -save -savePfx trpldev```

The default settings in mr_clust_embed.lua reflect those used in our final experiments (and so, for instance, both dev and train will be used as training data), but see the file for additional options and documentation.

You can download bzipped trained full model components from https://drive.google.com/folderview?id=0B1ytQXPDuw7OYUcwSEVPRjFEM00&usp=sharing , where the relevant files are trpldev-mce-700-200.model-na.bz2, trpldev-mce-700-200.model-pw.bz2, and trpldev-mce-700-200.model-lstm.bz2

## Predicting with Saved Models
If you've trained (or downloaded) full model components, you can make predictions as follows:

- If they don't exist, create the directories nn/bps/ and nn/conllouts/ .
- Run ```th mr_clust_embed.lua -gpuid 0 -loadAndPredict -pwDevFeatPrefix test_small -anaDevFeatPrefix test_small -savedPWNetFi models/trpldev-mce-700-200.model-pw -savedNANetFi models/trpldev-mce-700-200.model-na -savedLSTMFi models/trpldevdup-mce-700-200.model-lstm```
- The above will create a back-pointer file in bps/ . Suppose the file is called bps/xyzdev.bps . Then to generate a CoNLL output file, run ```../modifiedBCS/WriteCoNLLPreds.sh bps bps/xyzdev.bps conllouts ../flat_test_2012/ ../gender.data```
    - N.B. You may need to modify the paths to the jar files on the second line of modifiedBCS/WriteCoNLLPreds.sh to get this to work
- The resulting output file (in conllouts/) can now be scored using the standard CoNLL scorer.

Training as in the previous sub-section and evaluating as above should produce results very close to those in the NAACL paper, and probably a bit better. After re-training the cleaned-up and re-factored version in this repo, I got P/R/F scores of:

MUC: 77.14/70.12/73.46

BCUB: 66.43/57.47/61.62

CEAFe: 62.29/54.01/57.85

CoNLL: 64.31

## Training the ACL (non-cluster) Model
The mention-ranking model from the ACL paper has been re-implemented and considerably simplified in vanilla_mr.lua. It can be run as follows:

```th vanilla_mr.lua -gpuid 0 -PT```

Unlike the original ACL implementation, this implementation is easy to run on a GPU, and with the new, even-smaller feature-set it should do at least as well. 

## Copyright
Copyright (c) 2016 Sam Wiseman. All Rights Reserved.

## License
The code in this repository is covered by a GNU GPL License. See LICENSE.txt.

