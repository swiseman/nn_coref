## Overview

This directory contains the code necessary for extracting features and "oracle predicted clusters" (which are used as supervision) from the English CoNLL data. 

We use code written on top of the Berkeley Coref System (BCS) v1.1 (see http://nlp.cs.berkeley.edu/projects/coref.shtml) to extract features, and so we have included the BCS v1.1 code along with its license and dependencies here. The BCS code is in src/main/java/edu/berkeley/* and any additional code we have added is in src/main/java/edu/harvard/* .

## Compilation

Although we provide a pre-compiled jar ("moarcoref-assembly-1.jar") in the modifiedBCS/ directory, you can use sbt to re-compile the scala and java source. After downloading sbt (www.scala-sbt.org), simply type
  
```
sbt assembly
```

from inside the modifiedBCS/ directory, which will produce a runnable jar in the target/ subdirectory.

## Data Prerequisites

To extract features you will need the CoNLL 2012 English train, development, and test data, as well as the number and gender data that goes along with it. See http://conll.cemantix.org/2012/data.html for instructions on downloading and extracting it. 

BCS expects the CoNLL data to be in a flattened directories, so that all train, development, and test files are in flat train, development, and test directories (resp.).  If you've extracted the CoNLL data into a top-level directory called conll-2012/, you can create a flattened train directory flat_train_2012/ using the following python code:

```python
import subprocess
import shutil
import os

def flatten(root_dir,flat_dir,file_suf="auto_conll"):
    if not os.path.exists(flat_dir):
        os.makedirs(flat_dir)
    
    matches = subprocess.check_output("find %s -name *%s" % (root_dir,file_suf),shell=True)
    matches = matches.split('\n')[:-1]
    for match in matches:
        match_fields = match.split('/')
        shutil.copyfile(match, os.path.join(flat_dir,match_fields[-4]+"_"+match_fields[-1]))


flatten("conll-2012/v4/data/train/data/english", "flat_train_2012")
```

The same goes for creating flattened development and test directories.

You will also need the list of animate and inanimate unigrams used by the Stanford Coref system. These can be found in the Stanford CoreNLP models jar under edu.stanford.nlp.models.dcoref .

## Running

To extract the features described in the (NAACL) paper, first create a directory to store log files (say, `execdir'), and then type the following

```
java -jar -Xmx30g modifiedBCS/target/scala-2.11/moarcoref-assembly-1.jar ++modifiedBCS/base.conf -execDir execdir -numberGenderData gender.data -animacyPath animate.unigrams.txt -inanimacyPath inanimate.unigrams.txt -trainPath flat_train_2012 -devPath flat_dev_2012 -testPath flat_test_2012  -mode SMALLER -conjType NONE -pairwiseFeats FINAL+MOARANAPH+MOARPW
```

The above assumes the gender and animacy files are in the current directory, and that the flattened CoNLL directories are flat_train_2012/, flat_dev_2012/, and flat_test_2012/. 

The argument to pairwiseFeats specifies which features to extract. The argument `FINAL+MOARANAPH+MOARPW` corresponds to the features described in the paper.

There are additional options described in edu.harvard.nlp.moarcoref.MiniDriver.java.

## Output Generated

Running as above should give you 10 files, as follows:

 - SMALL-FINAL+MOARANAPH+MOARPW-anaph\[Train|Dev|Test\]Feats.txt
 
     Anaphoricity features. These files put each document on its own line, with each line having the following format:
     
     ```
     num_mentions_in_doc|ment_0_feat_0 ment_0_feat_1 ...|ment_n_feat_0 ...
     ``` 
     
     where n is the number of mentions in the document.
     
 - SMALL-FINAL+MOARANAPH+MOARPW-pw\[Train|Dev|Test\]Feats.txt
 
     Pairwise features. These files put each document on its own line, with each line having the following format:
     
     ```
     num_mentions_in_doc|ment_0_ant_0_feat_0 ment_0_ant_0_feat_1 ...|ment_1_ant_0_feat_0 ment_1_ant_0_feat_1 ...|...|ment_n_ant_n_feat_0 ...
     ```
     
     As such, there are n(n+1)/2 cells containing features on each line (one for each pair of mention-antecedent pairs plus self-link mention-mention pairs), and n(n+1)/2+1 cells in total, because the first cell contains the number of mentions. Since the pairwise features do not make sense for the self-link mention-mention pairs, we simply insert a dummy integer in the corresponding cell. 
     
 - SMALL-FINAL+MOARANAPH+MOARPW-\[anaph|pw\]Mapping.txt
     
     A file mapping feature index numbers to feature descriptions. Each feature is on its own line, and the format is:
     
     ```
     feature_idx : feature_description
     ```
     
 - SMALL\[Train|Dev\]OPCs.txt
 
     Oracle Predicted Clustering files. These are the clusterings induced by the true gold clusters on the mentions extracted by the automatic mention extractor, and they constitute the supervision for this task. Again each document is on its own line, where each line contains clusters separated by a `|`, and the mention indices within a cluster are separated by a space, and are in ascending order. For example the following line
     
     ```
      0|1 2 4|3
     ```
     
     indicates that there are 3 clusters over 5 mentions, with the first and third cluster just containing the first and fourth mentions (resp.), and the second cluster containing the 2nd, 3rd, and 5th mentions.

## System Requirements

In addition to sbt you will need java. When running without any real memory restrictions, feature extraction requires around 30GB of RAM; it's likely that you can get away with a bit less than this, however.

