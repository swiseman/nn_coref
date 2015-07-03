package edu.berkeley.nlp.coref.preprocess;

import edu.berkeley.nlp.futile.util.Logger;
import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;


public class NerDriver implements Runnable {
  @Option(gloss = "")
  public static Mode mode = Mode.TRAIN;

  @Option(gloss = "Path to read/write the model")
  public static String modelPath = "";

  // TRAINING_OPTIONS
  @Option(gloss = "Path to CoNLL training set")
  public static String trainPath = "";
  @Option(gloss = "Training set size, -1 for all")
  public static int trainSize = -1;
  @Option(gloss = "Path to CoNLL test set")
  public static String testPath = "";
  @Option(gloss = "Test set size, -1 for all")
  public static int testSize = -1;
  
  public static enum Mode {
    TRAIN, RUN;
  }
  
  public static void main(String[] args) {
    NerDriver main = new NerDriver();
    Execution.run(args, main); // add .class here if that class should receive command-line args
  }
  
  public void run() {
    Logger.setFig();
    switch (mode) {
      case TRAIN: NerSystem.trainNerSystem();
        break;
      case RUN:
        // Read trees
//        PennTreeReader.berkeleyParserBadTree
        // Extract words and POS
        
        // 
        break;
    }
  }
}
