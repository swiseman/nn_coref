package edu.berkeley.nlp.coref.preprocess;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.PCFGLA.CoarseToFineMaxRuleParser;
import edu.berkeley.nlp.PCFGLA.Grammar;
import edu.berkeley.nlp.PCFGLA.Lexicon;
import edu.berkeley.nlp.PCFGLA.ParserData;
import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.coref.ConllDocReader;
import edu.berkeley.nlp.coref.lang.Language;
import edu.berkeley.nlp.futile.fig.basic.IOUtils;
import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;
import edu.berkeley.nlp.futile.util.Logger;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;


public class PreprocessingDriver implements Runnable {
  @Option(gloss = "")
  public static Mode mode = Mode.RAW_TEXT;

  @Option(gloss = "Path to read the sentence splitter model from")
  public static String sentenceSplitterModelPath = "models/sentsplit.txt.gz";
  @Option(gloss = "Path to read the Berkeley Parser grammar from")
  public static String grammarPath = "models/eng_sm6.gr";
  @Option(gloss = "Path to read a backoff grammar from")
  public static String backoffGrammarPath = "models/eng_sm1.gr";
  @Option(gloss = "Path to read the NER model from")
  public static String nerModelPath = "models/ner.txt.gz";
  
  @Option(gloss = "Raw text or CoNLL input directory")
  public static String inputDir = "";
  @Option(gloss = "CoNLL annotation output directory")
  public static String outputDir = "";
  @Option(gloss = "Skip sentence splitting entirely.")
  public static boolean skipSentenceSplitting = false;
  @Option(gloss = "Respect line breaks for sentence segmentation (i.e. a new line always means a new sentence). False by default.")
  public static boolean respectInputLineBreaks = false;
  @Option(gloss = "Respect two consecutive line breaks for sentence segmentation (i.e. a blank line always means a new sentence). True by default.")
  public static boolean respectInputTwoLineBreaks = true;
  
  public static enum Mode {
    RAW_TEXT, REDO_CONLL;
  }
  
  public static void main(String[] args) {
    PreprocessingDriver main = new PreprocessingDriver();
    Execution.run(args, main); // add .class here if that class should receive command-line args
  }
  
  public void run() {
    Logger.setFig();
    Logger.logss("Loading sentence splitter");
    SentenceSplitter splitter = SentenceSplitter.loadSentenceSplitter(sentenceSplitterModelPath);
    Logger.logss("Loading parser");
    CoarseToFineMaxRuleParser parser = loadParser(grammarPath);
    Logger.logss("Loading backoff parser");
    CoarseToFineMaxRuleParser backoffParser = loadParser(backoffGrammarPath);
    Logger.logss("Loading NER system");
    NerSystem nerSystem = NerSystem.loadNerSystem(nerModelPath);
    if (!inputDir.isEmpty() && !outputDir.isEmpty() && !inputDir.equals(outputDir)) {
      if (mode == Mode.RAW_TEXT) {
        for (File inputFile : new File(inputDir).listFiles()) {
          processDocument(splitter, parser, backoffParser, nerSystem, inputDir + "/" + inputFile.getName(), outputDir + "/" + inputFile.getName());
        }
      } else {
        ConllDocReader docReader = new ConllDocReader(Language.ENGLISH);
        for (File inputFile : new File(inputDir).listFiles()) {
          Reprocessor.redoConllDocument(parser, backoffParser, nerSystem, docReader, inputDir + "/" + inputFile.getName(), outputDir + "/" + inputFile.getName());
        }
      }
    } else {
      Logger.logss("Need to provide either a distinct inputPath/outputPath pair or a distinct inputDir/outputDir");
    }
  }
  
  // TODO: Redo this to use edu.berkeley.nlp.coref.ConllDocWriter.writeIncompleteConllDoc
  public static void processDocument(SentenceSplitter splitter, CoarseToFineMaxRuleParser parser, CoarseToFineMaxRuleParser backoffParser, NerSystem nerSystem, String inputPath, String outputPath) {
    String[] lines = IOUtils.readLinesHard(inputPath).toArray(new String[0]);
    String[] canonicalizedParagraphs = splitter.formCanonicalizedParagraphs(lines, respectInputLineBreaks, respectInputTwoLineBreaks);
    String[] sentences = null;
    if (skipSentenceSplitting) {
      sentences = canonicalizedParagraphs;
    } else {
      sentences = splitter.splitSentences(canonicalizedParagraphs);
    }
    String[][] tokenizedSentences = splitter.tokenize(sentences);
    Logger.logss("Document " + inputPath + " contains " + lines.length + " lines and " + tokenizedSentences.length + " sentences");
    PrintWriter writer = IOUtils.openOutHard(outputPath);
    writer.println("#begin document (" + inputPath + "); part 000");
    for (String[] tokenizedSentence: tokenizedSentences) {
      Tree<String> parse = parse(parser, backoffParser, Arrays.asList(tokenizedSentence));
      if (parse.getYield().size() != tokenizedSentence.length) {
        Logger.logss("WARNING: couldn't parse sentence, dropping it: " + Arrays.toString(tokenizedSentence));
        Logger.logss("  (This will be fixed to backing off to an X-bar grammar in a future release)");
      } else {
        String[] posTags = new String[tokenizedSentence.length];
        List<String> preterminals = parse.getPreTerminalYield();
        for (int i = 0; i < preterminals.size(); i++) {
          posTags[i] = preterminals.get(i);
        }
        String[] nerBioLabels = nerSystem.runNerSystem(tokenizedSentence, posTags);
        String[] conllLines = renderConllLines(inputPath, 0, tokenizedSentence, posTags, parse, nerBioLabels);
        for (String conllLine : conllLines) {
          writer.println(conllLine);
        }
        writer.println();
      }
    }
    writer.println("#end document");
    writer.close();
    Logger.logss("Processed document " + inputPath + " and wrote result to " + outputPath);
  }

  public static String[] renderConllLines(String docName, int partNo, String[] words, String[] pos, Tree<String> parse, String[] nerBioLabels) {
    assert words.length == pos.length;
    assert words.length == parse.getYield().size();
    assert words.length == nerBioLabels.length;
    // PARSE PROCESSING
    String[] parseBits = computeParseBits(parse);
    String[] nerBits = computeNerBits(nerBioLabels);
    String[] conllLines = new String[words.length];
    for (int i = 0; i < words.length; i++) {
      conllLines[i] = docName + "\t" + partNo + "\t" + i + "\t" + words[i] + "\t" + pos[i] + "\t" + parseBits[i] +
          "\t-\t-\t-\t-\t" + nerBits[i] + "\t-";
    }
    return conllLines;
  }
  
  public static String[] computeParseBits(Tree<String> parse) {
    String parseBitsConcat = parseTraversalHelper(parse);
    String[] bitsSplitAtStars = parseBitsConcat.split("\\*");
    for (int i = 1; i < bitsSplitAtStars.length; i++) {
      int firstIndexOfNonCloseParen = 0;
      while (firstIndexOfNonCloseParen < bitsSplitAtStars[i].length() && bitsSplitAtStars[i].charAt(firstIndexOfNonCloseParen) == ')') {
        firstIndexOfNonCloseParen++;
      }
      bitsSplitAtStars[i-1] += "*" + bitsSplitAtStars[i].substring(0, firstIndexOfNonCloseParen);
      bitsSplitAtStars[i] = bitsSplitAtStars[i].substring(firstIndexOfNonCloseParen);
    }
    assert bitsSplitAtStars[bitsSplitAtStars.length - 1].isEmpty();
    return bitsSplitAtStars;
  }

  private static String parseTraversalHelper(Tree<String> currTree) {
    if (currTree.isPreTerminal()) {
      return "*";
    } else {
      String childrenConcat = "";
      for (Tree<String> child : currTree.getChildren()) {
        childrenConcat += parseTraversalHelper(child);
      }
      String label = currTree.getLabel();
      if (label.equals("ROOT")) {
        label = "TOP";
      }
      return "(" + label + childrenConcat + ")";
    }
  }
  
  public static String[] computeNerBits(String[] nerBioLabels) {
    int size = nerBioLabels.length;
    String[] nerBits = new String[size];
    boolean inNer = false;
    for (int i = 0; i < size; i++) {
      if (nerBioLabels[i].startsWith("B")) {
        String nerType = "MISC";
        if (nerBioLabels[i].contains("-")) {
          nerType = nerBioLabels[i].substring(nerBioLabels[i].indexOf("-") + 1);
        }
        if (i == size - 1 || !nerBioLabels[i+1].startsWith("I")) {
          nerBits[i] = "(" + nerType + ")";
          inNer = false;
        } else {
          nerBits[i] = "(" + nerType + "*";
          inNer = true;
        }
      } else if (nerBioLabels[i].startsWith("I")) {
        assert inNer;
        if (i == size - 1 || !nerBioLabels[i+1].startsWith("I")) {
          nerBits[i] = "*)";
          inNer = false;
        } else {
          nerBits[i] = "*";
        }
      } else {
        nerBits[i] = "*";
        inNer = false;
      }
    }
    return nerBits;
  }
  
  public static CoarseToFineMaxRuleParser loadParser(String grFileName) {
    String inFileName = grFileName;
    ParserData pData = ParserData.Load(inFileName);
    if (pData == null) {
      System.out.println("Failed to load grammar from file " + inFileName);
      System.exit(1);
    }
    Grammar grammar = pData.getGrammar();
    Lexicon lexicon = pData.getLexicon();
    Numberer.setNumberers(pData.getNumbs());
    // Defaults from edu.berkeley.nlp.PCFGLA.BerkeleyParser
    double threshold = 1.0;
    boolean viterbi = false;
    boolean substates = false;
    boolean scores = false;
    boolean accurate = false;
    boolean variational = false;
    CoarseToFineMaxRuleParser parser = new CoarseToFineMaxRuleParser(grammar, lexicon,
          threshold, -1, viterbi, substates,
          scores, accurate, variational, true,
          true);
    parser.binarization = pData.getBinarization();
    return parser;
  }
  
  public static Tree<String> parse(CoarseToFineMaxRuleParser parser, CoarseToFineMaxRuleParser backoffParser, List<String> sentence) {
    int maxLength = 200;
    boolean goodParseFound = false;
    Tree<String> parsedTree = null;
    List<String> posTags = null;
    if (sentence.size() <= maxLength) {
      parsedTree = parser.getBestConstrainedParse(sentence, posTags, null);
      goodParseFound = parsedTree.getYield().size() == sentence.size();
    }
    if (!goodParseFound && backoffParser != null) {
      Logger.logss("Using backoff parser on sentence: " + sentence.toString());
      parsedTree = backoffParser.getBestConstrainedParse(sentence, posTags, null);
      goodParseFound = parsedTree.getYield().size() == sentence.size();
      if (!goodParseFound) {
        Logger.logss("WARNING: Backoff parser failed on sentence: " + sentence.toString());
      }
    }
    // Debinarize
    boolean keepFunctionLabels = false;
    parsedTree = TreeAnnotations.unAnnotateTree(parsedTree, keepFunctionLabels);
    return parsedTree;
  }
}
