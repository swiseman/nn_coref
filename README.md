# nn_coref
Neural Coref Models, as described in 
["Learning Global Features for Coreference Resolution"](http://nlp.seas.harvard.edu/papers/corefmain.pdf), Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber, NAACL 2016,
and
["Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution"](http://people.seas.harvard.edu/~srush/acl15.pdf), Sam Wiseman, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. ACL 2015.

For questions/concerns/bugs please contact swiseman@seas.harvard.edu.


## Overview

To keep things simple, all the ACL code is now in a different branch. This README will cover duplicating the NAACL 2016 results.

## Generating Features
First see the README in the modifiedBCS/ directory for instructions on setting up the data, and compiling the Scala feature and mention extractor.

Then, run

``` java -jar -Xmx30g modifiedBCS/target/scala-2.11/moarcoref-assembly-1.jar ++modifiedBCS/base.conf -execDir execdir -numberGenderData gender.data -animacyPath animate.unigrams.txt -inanimacyPath inanimate.unigrams.txt -trainPath flat_train_2012 -devPath flat_dev_2012 -testPath flat_test_2012  -mode SMALLER -conjType NONE -pairwiseFeats FINAL+MOARANAPH+MOARPW```

to generate text feature files.

To convert text feature files into hdf5 (to be consumed by Torch), run

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt train_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt dev_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-anaphTestFeats.txt test_small ana -n 4 -r 14215```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwTrainFeats.txt train_small pw -n 4 -r 28394```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwDevFeats.txt dev_small pw -n 4 -r 28394```

```python text_feats_to_hdf5_replacezero.py SMALL-FINAL+MOARANAPH+MOARPW-pwTestFeats.txt test_small pw -n 4 -r 28394```


## Running Experiments
The script run_experiments.py contains code and commands for duplicating our experiments, from feature extraction from the CoNLL files through training and evaluation. To run this script, you need to have extracted the CoNLL files from OntoNotes using the instructions at http://conll.cemantix.org/2012/data.html, which should give you a single top-level directory containing a hierarchy of CoNLL files.

## Pre-extracted Features/Saved Models
Pre-extracted Basic+ features (for train, dev, and test) can be downloaded here: https://drive.google.com/folderview?id=0B1ytQXPDuw7OYng1SGhFR0hRcnM&usp=sharing

Saved Models (including antecedent and anaphoricity subtask networks, and g1 and g2 pre-trained networks) can be downloaded here: https://drive.google.com/folderview?id=0B1ytQXPDuw7OeU5ENnZXS0JsNGs&usp=sharing

Both the features and saved models are bzipped, and must be bunzipped before you can use them.

## Copyright
Copyright (c) 2015 Sam Wiseman. All Rights Reserved.

## License
The code in this repository is covered a GNU GPL License. See LICENSE.txt.

