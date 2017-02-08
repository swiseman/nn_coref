package edu.harvard.nlp.moarcoref;

import edu.berkeley.nlp.coref.lang.Language;
import edu.berkeley.nlp.futile.util.Logger;
import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;
import edu.berkeley.nlp.coref.ConjType;

/*
 * A minimal version of BCS's Driver.java
 */
public class MiniDriver implements Runnable {
  
  @Option(gloss = "Which experiment to run?")
  public static Mode mode = Mode.SMALLER;
  @Option(gloss = "Language choice")
  public static Language lang = Language.ENGLISH;
  
  // DATA AND PATHS
  @Option(gloss = "Path to number/gender data")
  public static String numberGenderDataPath = "gender.data";
  @Option(gloss = "Path to Stanford Coref's animate unigrams")
  public static String animacyPath = "animate.unigrams.txt";
  @Option(gloss = "Path to Stanford Coref's inanimate unigrams")
  public static String inanimacyPath = "inanimate.unigrams.txt";  
  @Option(gloss = "Path to training set")
  public static String trainPath = "flat_train_2012";
  @Option(gloss = "Training set size, -1 for all")
  public static int trainSize = -1;
  @Option(gloss = "Path to dev set")
  public static String devPath = "flat_dev_2012";
  @Option(gloss = "Dev set size, -1 for all")
  public static int devSize = -1;  
  @Option(gloss = "Path to test set")
  public static String testPath = "flat_test_2012";
  @Option(gloss = "Test set size, -1 for all")
  public static int testSize = -1;
  @Option(gloss = "Suffix to use for documents")
  public static String docSuffix = "auto_conll";
  @Option(gloss = "Randomize the order of train documents")
  public static boolean randomizeTrain = true;

  @Option(gloss = "True if we should train on the documents with gold annotations, false if we should use auto annotations")
  public static boolean trainOnGold = false;
  @Option(gloss = "Use gold mentions.")
  public static boolean useGoldMentions = false;
 
  @Option(gloss = "Features to use; default is SURFACE, write \"+FINAL\" for FINAL")
  public static String pairwiseFeats = "";
  @Option(gloss = "Conjunction type")
  public static ConjType conjType = ConjType.CANONICAL;
  @Option(gloss = "Cutoff below which lexical features fire POS tags instead")
  public static int lexicalFeatCutoff = 20;
  @Option(gloss = "Cutoff below which bilexical features fire backoff indicator feature")
  public static int bilexicalFeatCutoff = 10;  
  
  
  public static enum Mode {
    SMALLER;
  }
  
  public static void main(String[] args) {
    MiniDriver main = new MiniDriver();
    Execution.run(args, main); // add .class here if that class should receive command-line args
  }
  
  public void run() {
    Logger.setFig();
    if (mode.equals(Mode.SMALLER)) {
        FeatureExtractor.writeSeparatedFeatsAndOraclePredClustering(true);
    } else {
    	FeatureExtractor.writeSeparatedFeatsAndOraclePredClustering(false);
    }
  }
}
