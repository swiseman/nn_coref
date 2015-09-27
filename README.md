# nn_coref
Neural Coref Models, as described in ["Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution"](http://people.seas.harvard.edu/~srush/acl15.pdf), Sam Wiseman, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. ACL 2015.

For questions/concerns/bugs please contact swiseman@seas.harvard.edu.

## Overview

This directory contains all the code necessary for duplicating our experiments. In particular, the modifiedBCS/ directory contains code for extracting features and for printing out predictions in CoNLL format, and the nn/ directory contains code for training models, saving them, and writing their predictions out. It also contains code for converting text feature files into hdf5 format, which is assumed by the nn/ models. Each of these subdirectories contains its own README; refer there for details. Finally, the run_experiments.py script in the current directory should reproduce our results. See it also for details on running the various pieces of the overall system. 

## Running Experiments
The script run_experiments.py contains code and commands for duplicating our experiments, from feature extraction from the CoNLL files through training and evaluation. To run this script, you need to have extracted the CoNLL files from OntoNotes using the instructions at http://conll.cemantix.org/2012/data.html, which should give you a single top-level directory containing a hierarchy of CoNLL files.

## Pre-extracted Features/Saved Models
Pre-extracted Basic+ features (for train, dev, and test) in can be downloaded here: https://drive.google.com/folderview?id=0B1ytQXPDuw7OYng1SGhFR0hRcnM&usp=sharing

Saved Models (including antecedent and anaphoricity subtask networks, and g1 and g2 pre-trained networks) can be downloaded here: https://drive.google.com/folderview?id=0B1ytQXPDuw7OeU5ENnZXS0JsNGs&usp=sharing

Both the features and saved models are bzipped, and must be bunzipped before you can use them.

## Copyright
Copyright (c) 2015 Sam Wiseman. All Rights Reserved.

## License
The code in the modifiedBCS/ directory derives from the Berkeley Coreference System (http://nlp.cs.berkeley.edu/projects/coref.shtml), and is thus covered by the GNU GPL License (v3). See modifiedBCS/LICENSE.txt. The code in reference-coreference-scorers/ has been entirely copied from the official scorer repo (https://github.com/conll/reference-coreference-scorers), and is governed by its own GNU GPL License.

All other code in this repository is covered by a non-commercial use license. The text of this non-commercial use license can be found in the nn/ subdirectory.
