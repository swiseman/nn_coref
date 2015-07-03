package edu.berkeley.nlp.coref;

import edu.berkeley.nlp.coref.lang.Language;
import edu.berkeley.nlp.coref.sem.QueryCountCollector;
import edu.berkeley.nlp.futile.util.Logger;
import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;


public class Driver implements Runnable {
  
  @Option(gloss = "Which experiment to run?")
  public static Mode mode = Mode.TRAIN_EVALUATE;
  @Option(gloss = "Language choice")
  public static Language lang = Language.ENGLISH;
  
  // DATA AND PATHS
  @Option(gloss = "Path to CoNLL evaluation script")
  public static String conllEvalScriptPath = "scorer/v5/scorer.pl";
  @Option(gloss = "Path to number/gender data")
  public static String numberGenderDataPath = "data/gender.data";
  @Option(gloss = "Path to WordNet")
  public static String wordNetPath = "data/WNdb-3.0/dict";
  
  @Option(gloss = "Path to training set")
  public static String trainPath = "";
  @Option(gloss = "Training set size, -1 for all")
  public static int trainSize = -1;
  @Option(gloss = "Path to test set")
  public static String testPath = "";
  @Option(gloss = "Test set size, -1 for all")
  public static int testSize = -1;
  @Option(gloss = "Suffix to use for documents")
  public static String docSuffix = "auto_conll";
  @Option(gloss = "Randomize the order of train documents")
  public static boolean randomizeTrain = true;

  @Option(gloss = "Path to read/write the model")
  public static String modelPath = "";
  @Option(gloss = "Path to write system output to")
  public static String outputPath = "";
  @Option(gloss = "Directory to write output CoNLL files to when using the scorer. If blank, uses the default temp directory and deletes them after. " +
     "You might want this because calling the scorer forks the process and may give an out-of-memory error, " +
     "so this is some insurance that at least you'll have your output.")
  public static String conllOutputDir = "";

  @Option(gloss = "Use NER in the system or ignore it?")
  public static boolean useNer = true;
  @Option(gloss = "True if we should train on the documents with gold annotations, false if we should use auto annotations")
  public static boolean trainOnGold = false;
  @Option(gloss = "Use gold mentions.")
  public static boolean useGoldMentions = false;
  @Option(gloss = "Can toggle whether written output is filtered for singletons or not")
  public static boolean doConllPostprocessing = true;
  @Option(gloss = "Include appositive mentions?")
  public static boolean includeAppositives = true;

  @Option(gloss = "Print per-document scores for bootstrap significance testing")
  public static boolean printSigSuffStats = false;
  
  // ORACLE OPTIONS
  @Option(gloss = "Use cheating clusters?")
  public static boolean cheat = false;
  @Option(gloss = "Number of cheating clusters")
  public static int numCheatingProperties = 3;
  @Option(gloss = "Domain size for each cheating cluster")
  public static int cheatingDomainSize = 5;
  
  // PHI FEATURE OPTIONS
  @Option(gloss = "Use phi-feature based clusters?")
  public static boolean phi = false;
  @Option(gloss = "Which phi cluster features to include (numb, gend, or nert)")
  public static String phiClusterFeatures = "";
  
  // TRAINING AND INFERENCE
  // These settings are reasonable and quite robust; you really shouldn't have
  // to tune the regularizer. I didn't find adjusting it useful even when going
  // from thousands to millions of features.
  @Option(gloss = "eta for Adagrad")
  public static double eta = 1.0;
  @Option(gloss = "Regularization constant (might be lambda or c depending on which algorithm is used)")
  public static double reg = 0.001;
  @Option(gloss = "Loss fcn to use")
  public static String lossFcn = "customLoss-0.1-3-1";
  @Option(gloss = "Loss fcn to use")
  public static String lossFcnSecondPass = "customLoss-0.1-3-1";
  @Option(gloss = "Number of iterations")
  public static int numItrs = 20;
  @Option(gloss = "Number of iterations")
  public static int numItrsSecondPass = 20;
  @Option(gloss = "Pruning strategy for coarse pass. No pruning by default")
  public static String pruningStrategy = "distance:10000:5000";
  @Option(gloss = "Pruning strategy for fine pass")
  public static String pruningStrategySecondPass = "c2flogratio:2";
  
  @Option(gloss = "Inference type")
  public static InferenceType inferenceType = InferenceType.PAIRWISE;
  @Option(gloss = "Features to use; default is SURFACE, write \"+FINAL\" for FINAL")
  public static String pairwiseFeats = "";
  @Option(gloss = "Features to use for the fine pass; default is SURFACE, write \"+FINAL\" for FINAL")
  public static String pairwiseFeatsSecondPass = "";
  @Option(gloss = "Conjunction type")
  public static ConjType conjType = ConjType.CANONICAL;
  @Option(gloss = "Conjunction type for the fine pass")
  public static ConjType conjTypeSecondPass = ConjType.CANONICAL;
  @Option(gloss = "Cutoff below which lexical features fire POS tags instead")
  public static int lexicalFeatCutoff = 20;
  @Option(gloss = "Decode with max or with left-to-right marginalization?")
  public static String decodeType = "basic";

  @Option(gloss = "See LexicalCountsBundle. Only used for an ablation for the talk.")
  public static String definitenessSubset = "";
  
  // BINARY OPTIONS
  @Option(gloss = "")
  public static double binaryLogThreshold = 0.0;
  @Option(gloss = "")
  public static boolean binaryNegateThreshold = false;
  @Option(gloss = "")
  public static String binaryClusterType = "TRANSITIVE_CLOSURE";
  @Option(gloss = "")
  public static double binaryNegativeClassWeight = 1.0;
  
  // RAHMAN/LOOPY OPTION
  @Option(gloss = "What kind of loopy featurization to use")
  public static String clusterFeats = "";
  
  // RAHMAN OPTIONS
  @Option(gloss = "")
  public static String rahmanTrainType = "goldusepred";
  
  // LOOPY OPTIONS
  @Option(gloss = "What to regularize the projected default weights towards")
  public static String projDefaultWeights = "agreeheavy";
  @Option(gloss = "Number of coref clusters to use (may be reduced from number of total clusters)")
  public static int corefClusters = 4;
  
  // ANALYSIS OPTIONS
  @Option(gloss = "Analyses to print: +purity, +discourse, +categorizer, +mistakes")
  public static String analysesToPrint = "";
  
  // COLLECT_COUNTS OPTIONS
  @Option(gloss = "Path to Google n-grams directory")
  public static String ngramRootDir = "";
  @Option(gloss = "Path to file where n-gram counts are written")
  public static String queryCountsFile = "";
  
  public static enum Mode {
    TRAIN, EVALUATE, PREDICT, TRAIN_EVALUATE, TRAIN_PREDICT, TWO_PASS, COLLECT_COUNTS;
  }
  
  public static enum InferenceType {
    BINARY, PAIRWISE, LOOPY, RAHMAN;
  }
  
  public static void main(String[] args) {
    Driver main = new Driver();
    Execution.run(args, main); // add .class here if that class should receive command-line args
  }
  
  public void run() {
    Logger.setFig();
    if (mode == Mode.TRAIN) {
      CorefSystem.runTrain(trainPath, trainSize, modelPath);
    } else if (mode == Mode.EVALUATE) {
      CorefSystem.runEvaluate(testPath, testSize, modelPath);
    } else if (mode == Mode.PREDICT) {
      CorefSystem.runPredict(testPath, testSize, modelPath, outputPath, doConllPostprocessing);
    } else if (mode == Mode.TRAIN_EVALUATE) {
      CorefSystem.runTrainEvaluate(trainPath, trainSize, testPath, testSize, modelPath);
    } else if (mode == Mode.TRAIN_PREDICT) {
      CorefSystem.runTrainPredict(trainPath, trainSize, testPath, testSize, modelPath, outputPath, doConllPostprocessing);
    } else if (mode == Mode.TWO_PASS) {
      CorefSystem.runNewOnlyTwoPass(trainPath, trainSize, testPath, testSize);
    } else if (mode == Mode.COLLECT_COUNTS) {
      QueryCountCollector.collectCounts(trainPath, trainSize, testPath, testSize, ngramRootDir, queryCountsFile);
    }
  }
}
