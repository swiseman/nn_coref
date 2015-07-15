#!/bin/python

import os
import shutil
import subprocess as sub
import sys
import argparse

"""
This script is intended to show how to reproduce the experiments in the paper,
and to provide a single executable for accomplishing this. It may be faster or
more convenient to instead run pieces individually (and to run the called
executables directly, rather than through subprocess.call(), as done below).
"""

def flatten(root_dir,flat_dir,file_suf="auto_conll"):
    if not os.path.exists(flat_dir):
        os.makedirs(flat_dir)
    matches = sub.check_output("find %s -name *%s" % (root_dir,file_suf),shell=True)
    matches = matches.split('\n')[:-1]
    for match in matches:
        match_fields = match.split('/')
        shutil.copyfile(match, os.path.join(flat_dir,match_fields[-4]+"_"+match_fields[-1]))

def make_flat_directories(opt):
    print "making flat_train_2012/"
    flatten("%s/v4/data/train/data/english" % opt.conll_dir, opt.flat_train)
    print "making flat_dev_2012/"
    flatten("%s/v4/data/development/data/english" % opt.conll_dir, opt.flat_dev)
    print "making flat_test_2012/"
    flatten("%s/v9/data/test/data/english" % opt.conll_dir, opt.flat_test)
    
def make_keyfile(root_dir, keyfi, file_suf="gold_conll"):
    matches = sub.check_output("find %s -name *%s" % (root_dir,file_suf),shell=True)
    matches = matches.split('\n')[:-1]
    sub.call("touch %s" % keyfi, shell=True)
    for match in matches:
        sub.call("cat %s >> %s" % (match,keyfi), shell=True)

def make_keyfiles(opt):
    make_keyfile("%s/v4/data/development/data/english" % opt.conll_dir, "dev.key")
    make_keyfile("%s/v4/data/test/data/english" % opt.conll_dir, "test.key")

def extract_modifiedBCS_features_and_OPCs(opt):
    if not os.path.exists(opt.execdir):
        os.makedirs(opt.execdir)
    mod_bcs_cmd = "java -jar -Xmx%dg modifiedBCS/moarcoref-assembly-1.jar"\
    " ++modifiedBCS/base.conf -execDir %s -numberGenderData %s -animacyPath %s"\
    " -inanimacyPath %s -trainPath %s -devPath %s -testPath %s -pairwiseFeats %s"\
    " -conjType NONE" % (opt.max_java_gb,opt.execdir,opt.ng_path,opt.an_path,
                        opt.inan_path,opt.flat_train,opt.flat_dev,opt.flat_test,opt.feats)
    print "running: %s" % mod_bcs_cmd
    sys.stdout.flush()
    sub.call(mod_bcs_cmd,shell=True)

def text_feats_to_hdf5(opt):
    """
    Assumes caller in nn/ directory.
    """
    tr_cmd = "python text_feats_to_hdf5.py -n %d ../NONE-%s-%s %s %s"
    dev_cmd = "python text_feats_to_hdf5.py -n %d -t ../NONE-%s-%s %s %s"
   
    # convert pairwise train features
    sub.call(tr_cmd % (opt.num_convert_procs,opt.feats,"pwTrainFeats.txt",
                                "train_"+opt.feat_set_abbrev,"pw"), shell=True)
    # convert pairwise dev features
    sub.call(dev_cmd % (opt.num_convert_procs,opt.feats,"pwDevFeats.txt",
                                 "dev_"+opt.feat_set_abbrev,"pw"), shell=True)
    # convert pairwise test features
    sub.call(dev_cmd % (opt.num_convert_procs,opt.feats,"pwTestFeats.txt",
                                "test_"+opt.feat_set_abbrev,"pw"), shell=True)

    # convert anaphoricity train features
    sub.call(tr_cmd % (opt.num_convert_procs,opt.feats,"anaphTrainFeats.txt",
                                "train_"+opt.feat_set_abbrev,"ana"), shell=True)
    # convert anaphoricity dev features
    sub.call(dev_cmd % (opt.num_convert_procs,opt.feats,"anaphDevFeats.txt",
                                 "dev_"+opt.feat_set_abbrev,"ana"), shell=True)
    # convert anaphoricity test features
    sub.call(dev_cmd % (opt.num_convert_procs,opt.feats,"anaphTestFeats.txt",
                                 "test_"+opt.feat_set_abbrev,"ana"), shell=True)

def compile_lua_modules():
    """
    Assumes caller in nn/ directory.
    """
    sub.call("luarocks make rocks/cr-scm-1.rockspec", shell=True)

def make_train_prereq_dirs():
    """
    Assumes caller in nn/ directory.
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("bps"):
        os.makedirs("bps")
    if not os.path.exists("conllouts"):
        os.makedirs("conllouts")

def pretrain(opt):
    """
    Assumes caller in nn/ directory.
    """
    anaph_pt_cmd = "th ana_model.lua -anaTrFeatPrefix %s -anaDevFeatPrefix %s"\
    " -save -savePrefix %s" % ("train_"+opt.feat_set_abbrev, "dev_"+opt.feat_set_abbrev, opt.feat_set_abbrev)
    print "running: %s" % anaph_pt_cmd
    sys.stdout.flush()
    sub.call(anaph_pt_cmd, shell=True)
    
    ante_pt_cmd = "th ante_model.lua -pwTrFeatPrefix %s -pwDevFeatPrefix %s"\
    " -save -savePrefix %s" % ("train_"+opt.feat_set_abbrev, "dev_"+opt.feat_set_abbrev, opt.feat_set_abbrev)
    print "running: %s" % ante_pt_cmd
    sys.stdout.flush()
    sub.call(ante_pt_cmd, shell=True)

def train(opt):
    tmplt = "th %s -pwTrFeatPrefix %s -anaTrFeatPrefix %s -eta1 %f -eta2 %f -lamb %f"\
    " -nEpochs %d %s -save -savePrefix %s"
    pt_path_str = "-antePTSerFile models/%s_700.model-pw-0.100000-0.000010"\
    " -anaphPTSerFile models/%s_128.model-na-0.100000-0.000010" % (opt.feat_set_abbrev,opt.feat_set_abbrev) 
    if opt.full_model == "g1":
        luafile = "full_g1_model.lua"
    else:
        luafile = "full_g2_model.lua"  

    if opt.no_pretrain:
        pt_opt_str = "-randomInit"        
    else:
        pt_opt_str = pt_path_str
        
    cmd = tmplt % (luafile, "train_"+opt.feat_set_abbrev, "train_"+opt.feat_set_abbrev,
                    opt.eta1, opt.eta2, opt.lamb,opt.num_epochs,pt_opt_str, opt.feat_set_abbrev)
    print "running: %s" % cmd
    sys.stdout.flush()
    sub.call(cmd,shell=True) 
        

def predict(opt, pfx="dev_"):
    """
    Assumes caller in nn/ directory.
    """
    tmplt = "th %s -pwDevFeatPrefix %s -anaDevFeatPrefix %s -loadAndPredict %s %s"
    if opt.full_model == "g1":
        modelname = "g1"
        luafile = "full_g1_model.lua"
        saved_model_pfx = opt.feat_set_abbrev
    else:
        modelname = "g2"
        luafile = "full_g2_model.lua"
        saved_model_pfx = opt.feat_set_abbrev + "_128"
    if opt.no_pretrain:
        initname = "RI"
    else:
        initname = "PT"
    load_pw = ("-pwFullSerFile models/%s-0.500000-1.200000-1.000000.model-full_%s_%s-pw" 
                % (saved_model_pfx, modelname, initname))
    load_ana = ("-anaFullSerFile models/%s-0.500000-1.200000-1.000000.model-full_%s_%s-na" 
                % (saved_model_pfx, modelname, initname))
    cmd = tmplt % (luafile, pfx+opt.feat_set_abbrev, pfx+opt.feat_set_abbrev, load_pw, load_ana)
    print "running: %s" % cmd
    sys.stdout.flush()
    sub.call(cmd,shell=True)

def get_conll_fmt_output(opt, dev=True):
    """
    Assumes caller in modifiedBCS/ directory.
    """
    print "getting predictions in CoNLL format..."
    sys.stdout.flush()
    sub.call("chmod +x WriteCoNLLPreds.sh", shell=True)
    if dev:
        sub.call("./WriteCoNLLPreds.sh ../nn/bps ../nn/conllouts ../%s ../gender.data" % opt.flat_dev, shell=True)
    else:
        sub.call("./WriteCoNLLPreds.sh ../nn/bps ../nn/conllouts ../%s ../gender.data" % opt.flat_test, shell=True)
    # sometimes these java procs refuse to die
    print "killing any remaining java processes..."
    sys.stdout.flush()
    sub.call("pkill java", shell=True)

def call_scorer_script(dev=True):
    """
    Assumes caller in main directory.
    """
    if dev:
        out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl"\
        " all dev.key nn/conllouts/load_and_pred.bps.out none", shell=True)
        print "conll scorer output:\n\n%s" % out
        sys.stdout.flush()
    else:
        out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl"\
        " all test.key nn/conllouts/load_and_pred.bps.out none", shell=True)
        print "conll scorer output:\n\n%s" % out  
        sys.stdout.flush()      
    
def main():
    """
    Make sure to run this script from the main nn_coref/ directory.
    """
    ap = argparse.ArgumentParser()
    # feature extraction opt
    ap.add_argument("-conll_dir", default="conll-2012", action="store", dest="conll_dir",
                          help="Top-level directory containing conll files")
    ap.add_argument("-flat_train", default="flat_train_2012", action="store", dest="flat_train",
                          help="Directory to store train files in")
    ap.add_argument("-flat_dev", default="flat_dev_2012", action="store", dest="flat_dev",
                          help="Directory to store development files in")
    ap.add_argument("-flat_test", default="flat_test_2012", action="store", dest="flat_test",
                          help="Directory to store test files in")
    ap.add_argument("-max_java_gb", type=int, default=30, action="store", dest="max_java_gb",
                          help="Max GB to use for BCS feature-extraction process")
    ap.add_argument("-execdir", default="execdir", action="store", dest="execdir",
                          help="Directory BCS feature-extraction logs to")
    ap.add_argument("-ng_path", default="gender.data", action="store", dest="ng_path",
                          help="Number/gender data file path")
    ap.add_argument("-an_path", default="animate.unigrams.txt", action="store", dest="an_path",
                          help="Animate unigrams file path")
    ap.add_argument("-inan_path", default="inanimate.unigrams.txt", action="store", dest="inan_path",
                          help="Inanimate unigrams file path")                          
    ap.add_argument("-feats", default="FINAL+MOARANAPH+MOARPW+bilexical", action="store", dest="feats",
                          help="Feature sets for BCS to extract; default corresponds to Basic+ features")
    # hdf5 conversion opt                          
    ap.add_argument("-num_convert_procs", type=int, default=30, action="store", dest="num_convert_procs",
                          help="Number of processes to use in converting text features to hdf5")
    ap.add_argument("-feat_set_abbrev", default="basicp", action="store", dest="feat_set_abbrev",
                          help="Identifier to prefix hdf5 features with")
    # training opt
    ap.add_argument("-no_pretrain", default=False, action="store_true", dest="no_pretrain",
                          help="Don't pretrain; randomly initialize full model")
    ap.add_argument("-model", choices=("g1","g2"), default="g1", action="store", dest="full_model",
                          help="Model to use for full coref task")
    ap.add_argument("-eta1", type=float, default=0.1, action="store", dest="eta1",
                          help="Learning rate for first layer of network")
    ap.add_argument("-eta2", type=float, default=0.001, action="store", dest="eta2",
                          help="Learning rate for second layer of network")
    ap.add_argument("-lamb", type=float, default=0.000001, action="store", dest="lamb",
                          help="l1 regularization coefficient")
    ap.add_argument("-num_epochs", type=int, default=14, action="store", dest="num_epochs",
                          help="Number of epochs to train")
    # eval opt
    ap.add_argument("-no_dev_eval", default=False, action="store_true", dest="no_dev_eval",
                          help="Don't evaluate on dev data")
    ap.add_argument("-eval_on_test", default=False, action="store_true", dest="eval_on_test",
                          help="Evaluate on test data")
    # pre-canned experimental settings from paper
    ap.add_argument("-run_g1_pt", default=False, action="store_true", dest="run_g1_pt",
                          help="Run pretrained g1 model on dev and test")
    ap.add_argument("-run_g1_ri", default=False, action="store_true", dest="run_g1_ri",
                          help="Run randomly initialized g1 model on dev")
    ap.add_argument("-run_g2_pt", default=False, action="store_true", dest="run_g2_pt",
                          help="Run pretrained g2 model on dev and test")
    ap.add_argument("-run_g2_ri", default=False, action="store_true", dest="run_g2_ri",
                          help="Run randomly initialized g2 model on dev")

    opt = ap.parse_args()
    
    # if we're running a pre-canned experiment, set everything up
    if opt.run_g1_pt:
        opt.full_model = "g1"
        opt.no_pretrain = False
        opt.eta1 = 0.1
        opt.eta2 = 0.001
        opt.lamb = 0.000001
        opt.num_epochs = 14
        opt.no_dev_eval = False
        opt.eval_on_test = True
    elif opt.run_g1_ri:
        opt.full_model = "g1"
        opt.no_pretrain = True
        opt.eta1 = 0.1
        opt.eta2 = 0.001
        opt.lamb = 0.000001
        opt.num_epochs = 11
        opt.no_dev_eval = False
        opt.eval_on_test = False
    elif opt.run_g2_pt:
        opt.full_model = "g2"
        opt.no_pretrain = False
        opt.eta1 = 0.1
        opt.eta2 = 0.001
        opt.lamb = 0.0001
        opt.num_epochs = 6
        opt.no_dev_eval = False
        opt.eval_on_test = True
    elif opt.run_g2_ri:
        opt.full_model = "g2"
        opt.no_pretrain = True
        opt.eta1 = 0.2
        opt.eta2 = 0.001
        opt.lamb = 0.00001
        opt.num_epochs = 7
        opt.no_dev_eval = False
        opt.eval_on_test = False
    
    make_flat_directories(opt)  
    make_keyfiles(opt)  
    extract_modifiedBCS_features_and_OPCs(opt)
    
    # switch to nn directory for training
    os.chdir("nn")
    
    # convert features into hdf5 so torch can read them more easily
    text_feats_to_hdf5(opt)
    
    # do training
    compile_lua_modules()
    make_train_prereq_dirs()  
    
    if not opt.no_pretrain:
        pretrain(opt)
              
    train(opt)
    
    if not opt.no_dev_eval:
        if not os.getcwd().endswith("/nn"):
            os.chdir("nn")
        predict(opt, pfx="dev_")
        os.chdir("../modifiedBCS")
        get_conll_fmt_output(opt, dev=True)
        os.chdir("..")
        call_scorer_script(dev=True)
        
    if opt.eval_on_test:
        if not os.getcwd().endswith("/nn"):
            os.chdir("nn")        
        predict(opt, pfx="test_")
        os.chdir("../modifiedBCS")
        get_conll_fmt_output(opt, dev=False)
        os.chdir("..")
        call_scorer_script(dev=False)        
    
if __name__ == "__main__":
    main()  