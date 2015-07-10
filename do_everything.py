#!/bin/python

import os
import shutil
import subprocess as sub

# This script is intended to show how to reproduce the experiments in the paper,
# and to provide a single executable for accomplishing this. It may be faster or
# more convenient to instead run pieces individually (and to run the called
# executables directly, rather than through subprocess.call(), as done below).

################## some constants you might want to change #####################
# feature extraction
CONLL_DIR = "conll-2012"
FLAT_TRAIN = "flat_train_2012"
FLAT_DEV = "flat_dev_2012"
FLAT_TEST = "flat_test_2012"
MAX_JAVA_GB = 30
EXECDIR = "execdir"
NG_PATH = "gender.data"
AN_PATH = "animate.unigrams.txt"
INAN_PATH = "inanimate.unigrams.txt"
FEATS = "FINAL+MOARANAPH+MOARPW+bilexical"

# hdf5 conversion
NUM_CONVERT_PROCS = 30
FEAT_SET_ABBREV = "basicp"

# training
PRETRAIN = True
FULL_MODEL = "g1"

# eval
EVAL_ON_DEV = True
EVAL_ON_TEST = True
################################################################################

def flatten(root_dir,flat_dir,file_suf="auto_conll"):
    if not os.path.exists(flat_dir):
        os.makedirs(flat_dir)
    matches = sub.check_output("find %s -name *%s" % (root_dir,file_suf),shell=True)
    matches = matches.split('\n')[:-1]
    for match in matches:
        match_fields = match.split('/')
        shutil.copyfile(match, os.path.join(flat_dir,match_fields[-4]+"_"+match_fields[-1]))

def make_flat_directories():
    print "making flat_train_2012/"
    flatten("%s/v4/data/train/data/english" % CONLL_DIR, FLAT_TRAIN)
    print "making flat_dev_2012/"
    flatten("%s/v4/data/development/data/english" % CONLL_DIR, FLAT_DEV)
    print "making flat_test_2012/"
    flatten("%s/v9/data/test/data/english" % CONLL_DIR, FLAT_TEST)
    
def make_keyfile(root_dir, keyfi, file_suf="gold_conll"):
    matches = sub.check_output("find %s -name *%s" % (root_dir,file_suf),shell=True)
    matches = matches.split('\n')[:-1]
    sub.call("touch %s" % keyfi, shell=True)
    for match in matches:
        sub.call("cat %s >> %s" % (match,keyfi), shell=True)

def make_keyfiles():
    make_keyfile("%s/v4/data/development/data/english" % CONLL_DIR, "dev.key")
    make_keyfile("%s/v4/data/test/data/english" % CONLL_DIR, "test.key")

def extract_modifiedBCS_features_and_OPCs():
    if not os.path.exists(EXECDIR):
        os.makedirs(EXECDIR)
    mod_bcs_cmd = "java -jar -Xmx%dg modifiedBCS/moarcoref-assembly-1.jar"
    " ++modifiedBCS/base.conf -execDir %s -numberGenderData %s -animacyPath %s"
    " -inanimacyPath %s -trainPath %s -devPath %s -testPath %s -pairwiseFeats %s"
    " -conjType NONE" % (MAX_JAVA_GB,EXECDIR,NG_PATH,AN_PATH,INAN_PATH,FLAT_TRAIN,FLAT_DEV,FLAT_TEST,FEATS)
    print "running: %s" % mod_bcs_cmd
    sub.call(mod_bcs_cmd,shell=True)

def text_feats_to_hdf5():
   tr_cmd = "python text_feats_to_hdf5.py -n %d %s-%s %s %s"
   dev_cmd = "python text_feats_to_hdf5.py -n %d -t %s-%s %s %s"
   
   # convert pairwise train features
   sub.call(tr_cmd % (NUM_CONVERT_PROCS,FEATS,"pwTrainFeats.txt","train_"+FEAT_SET_ABBREV,"pw"), shell=True)
   # convert pairwise dev features
   sub.call(dev_cmd % (NUM_CONVERT_PROCS,FEATS,"pwDevFeats.txt","dev_"+FEAT_SET_ABBREV,"pw"), shell=True)
   # convert pairwise test features
   sub.call(dev_cmd % (NUM_CONVERT_PROCS,FEATS,"pwTestFeats.txt","test_"+FEAT_SET_ABBREV,"pw"), shell=True)

   # convert anaphoricity train features
   sub.call(tr_cmd % (NUM_CONVERT_PROCS,FEATS,"anaphTrainFeats.txt","train_"+FEAT_SET_ABBREV,"ana"), shell=True)
   # convert anaphoricity dev features
   sub.call(dev_cmd % (NUM_CONVERT_PROCS,FEATS,"anaphDevFeats.txt","dev_"+FEAT_SET_ABBREV,"ana"), shell=True)
   # convert anaphoricity test features
   sub.call(dev_cmd % (NUM_CONVERT_PROCS,FEATS,"anaphTestFeats.txt","test_"+FEAT_SET_ABBREV,"ana"), shell=True)

def make_train_prereq_dirs():
    """
    assumes in nn/ directory
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("bps"):
        os.makedirs("bps")
    if not os.path.exists("conllouts"):
        os.makedirs("conllouts")

def pretrain():
    """
    assumes in nn/ directory
    """
    anaph_pt_cmd = "th ana_model.lua -anaTrFeatPfx %s -anaDevFeatPfx %s"
    " -save -savePfx %s" % ("train_"+FEAT_SET_ABBREV, "dev_"+FEAT_SET_ABBREV, FEAT_SET_ABBREV)
    print "running: %s" % anaph_pt_cmd
    sub.call(anaph_pt_cmd, shell=True)
    
    ante_pt_cmd = "th ante_model.lua -pwTrFeatPfx %s -pwDevFeatPfx %s"
    " -save -savePfx %s" % ("train_"+FEAT_SET_ABBREV, "dev_"+FEAT_SET_ABBREV, FEAT_SET_ABBREV)
    print "running: %s" % ante_pt_cmd
    sub.call(ante_pt_cmd, shell=True)

def train():
    tmplt = "th %s -pwTrFeatPfx %s -anaTrFeatPfx %s %s"
    pt_opts = "-antePTSerFi models/%s_700.model-pw-0.100000-0.000010"
    " -anaphPTSerFi models/%s_128.model-na-0.100000-0.000010" % (FEAT_SET_ABBREV)   
    cmd = None 
    if PRETRAIN:
        cmd = tmplt % ("full_g1_model.lua" if FULL_MODEL == "g1" else "full_g2_model.lua",
                      "train_"+FEAT_SET_ABBREV, "train_"+FEAT_SET_ABBREV, pt_opts)
        print "running: " % cmd
        sub.call(cmd,shell=True)
    else:
        cmd = tmplt % ("full_g1_model.lua" if FULL_MODEL == "g1" else "full_g2_model.lua",
                      "train_"+FEAT_SET_ABBREV, "train_"+FEAT_SET_ABBREV, "-random_init")
        print "running: " % cmd
        sub.call(cmd,shell=True) 

def predict(pfx="dev_"):
    tmplt = "th %s -pwDevFeatPfx %s -anaDevFeatPfx %s -load_and_predict %s %s"
    load_pw = ("-pwFullSerFi models/%s-0.500000-1.200000-1.000000.model-full_%s_%s-pw" 
                % (FEAT_SET_ABBREV,"g1" if FULL_MODEL == "g1" else "g2","PT" if PRETRAIN else "RI"))
    load_ana = ("-anaFullSerFi models/%s-0.500000-1.200000-1.000000.model-full_%s_%s-na" 
                % (FEAT_SET_ABBREV,"g1" if FULL_MODEL == "g1" else "g2","PT" if PRETRAIN else "RI"))
    cmd = tmplt % ("full_g1_model.lua" if FULL_MODEL == "g1" else "full_g2_model.lua",
                  pfx+FEAT_SET_ABBREV,pfx+FEAT_SET_ABBREV,load_pw,load_ana)
    print "running: " % cmd
    sub.call(cmd,shell=True)

def get_conll_fmt_output(dev=True):
    """
    assumes in modifiedBCS/ directory
    """
    if dev:
        sub.call("./WriteCoNLLPreds.sh ../nn/bps ../nn/conllouts ../%s ../gender.data" % FLAT_DEV, shell=True)
    else:
        sub.call("./WriteCoNLLPreds.sh ../nn/bps ../nn/conllouts ../%s ../gender.data" % FLAT_TEST, shell=True)

def call_scorer_script(dev=True):
    """
    assumes in main directory
    """
    if dev:
        out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl"
        " all dev.key nn/conllouts/load_and_pred.bps.out none", shell=True)
        print "conll scorer output:\n\n%s" % out
    else:
        out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl"
        " all test.key nn/conllouts/load_and_pred.bps.out none", shell=True)
        print "conll scorer output:\n\n%s" % out        
    
def main():
    """
    make sure to run this script from the main nn_coref/ directory
    """
    make_flat_directories()  
    make_keyfiles()  
    extract_modifiedBCS_features_and_OPCs()
    
    # switch to nn directory for training
    os.chdir("nn")
    
    # convert features into hdf5 so torch can read them more easily
    text_feats_to_hdf5()
    
    # do training
    make_train_prereq_dirs()  
    
    if PRETRAIN:
        pretrain()
              
    train()
    
    os.chdir("..")
    
    if EVAL_ON_DEV:
        predict(pfx="dev_")
        os.chdir("modifiedBCS")
        get_conll_fmt_output(dev=True)
        os.chdir("..")
        call_scorer_script(dev=True)
        
    if EVAL_ON_TEST:
        predict(pfx="test_")
        os.chdir("modifiedBCS")
        get_conll_fmt_output(dev=False)
        os.chdir("..")
        call_scorer_script(dev=False)        
    
if __name__ == "__main__":
    main()  