package edu.berkeley.nlp.coref
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.coref.Driver.InferenceType
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.basic.SysInfoUtils
import javax.management.Query
import edu.berkeley.nlp.coref.sem.QueryCountsBundle

object CorefSystem {
  
  def checkFileReachableForRead(file: String, msg: String) {
    if (file.isEmpty) {
      throw new RuntimeException("Undefined " + msg + "; must be defined for the mode you're running in");
    }
    if (!new File(file).exists()) {
      throw new RuntimeException(msg + " file/directory doesn't exist for read: " + file);
    }
  }
  def checkFileReachableForWrite(file: String, msg: String) {
    if (file.isEmpty) {
      throw new RuntimeException("Undefined " + msg + "; must be defined for the mode you're running in");
    }
    if (!new File(file).getParentFile().exists()) {
      throw new RuntimeException(msg + " file/directory couldn't be opened for write: " + file);
    }
  }
  
  def loadRawConllDocs(path: String, size: Int, gold: Boolean): Seq[ConllDoc] = {
    val suffix = if (gold) "gold_conll" else Driver.docSuffix;
    Logger.logss("Loading " + size + " docs from " + path + " ending with " + suffix);
    val files = new File(path).listFiles().filter(file => file.getAbsolutePath.endsWith(suffix));
    val reader = new ConllDocReader(Driver.lang);
    val docs = new ArrayBuffer[ConllDoc];
    var docCounter = 0;
    var fileIdx = 0;
    while (fileIdx < files.size && (size == -1 || docCounter < size)) {
      val newDocs = reader.readConllDocs(files(fileIdx).getAbsolutePath);
      docs ++= newDocs;
      docCounter += newDocs.size
      fileIdx += 1;
    }
    val numDocs = if (size == -1) docs.size else Math.min(size, files.size);
    Logger.logss(docs.size + " docs loaded from " + fileIdx + " files, retaining " + numDocs);
    if (docs.size == 0) {
      Logger.logss("WARNING: Zero docs loaded...double check your paths unless you meant for this happen");
    }
    val docsToUse = docs.slice(0, numDocs);
    
    docsToUse;
  }
  
  def loadCorefDocs(path: String, size: Int, numberGenderComputer: NumberGenderComputer, gold: Boolean): Seq[CorefDoc] = {
    val docs = loadRawConllDocs(path, size, gold);
    val assembler = CorefDocAssembler(Driver.lang, gold);
    val mentionPropertyComputer = new MentionPropertyComputer(numberGenderComputer);
    val corefDocs = docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    CorefDoc.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
  
  def saveModelFile(scorer: PairwiseScorer, modelPath: String) {
    try {
      val fileOut = new FileOutputStream(modelPath);
      val out = new ObjectOutputStream(fileOut);
      out.writeObject(scorer);
      Logger.logss("Model written to " + modelPath);
      out.close();
      fileOut.close();
    } catch {
      case e: Exception => throw new RuntimeException(e);
    }
  }
  
  def loadModelFile(modelPath: String): PairwiseScorer = {
    var scorer: PairwiseScorer = null;
    try {
      val file = new File(modelPath);
      val fileIn = new FileInputStream(file);
      val in = new ObjectInputStream(fileIn);
      scorer = in.readObject().asInstanceOf[PairwiseScorer]
      Logger.logss("Model read from " + modelPath);
      in.close();
      fileIn.close();
    } catch {
      case e: Exception => throw new RuntimeException(e);
    }
    scorer;
  }
  
  def preprocessDocsClusterInfo(allDocGraphs: Seq[DocumentGraph]) {
    // Store oracle information
    if (Driver.cheat) {
      val rng = new java.util.Random(0);
      for (i <- 0 until Driver.numCheatingProperties) {
        allDocGraphs.map(_.computeAndStoreCheatingPosteriors(Driver.cheatingDomainSize, rng));
      }
    }
    // Store phi features
    if (Driver.phi) {
      val useNum = Driver.phiClusterFeatures.contains("numb");
      val useGender = Driver.phiClusterFeatures.contains("gend");
      val useNert = Driver.phiClusterFeatures.contains("nert");
      allDocGraphs.map(_.computeAndStorePhiPosteriors(useNum, useGender, useNert));
    }
  }
  
  def runTrainEvaluate(trainPath: String, trainSize: Int, devPath: String, devSize: Int, modelPath: String) {
    checkFileReachableForRead(Driver.conllEvalScriptPath, "conllEvalScriptPath");
    checkFileReachableForRead(devPath, "testPath");
    val scorer = runTrain(trainPath, trainSize);
    if (!modelPath.isEmpty) {
      saveModelFile(scorer, modelPath);
    }
    runEvaluate(devPath, devSize, scorer);
  }
  
  def runTrainPredict(trainPath: String, trainSize: Int, devPath: String, devSize: Int, modelPath: String, outPath: String, doConllPostprocessing: Boolean) {
    checkFileReachableForRead(devPath, "testPath");
    checkFileReachableForWrite(outPath, "outputPath");
    val scorer = runTrain(trainPath, trainSize);
    if (!modelPath.isEmpty) {
      saveModelFile(scorer, modelPath);
    }
    runPredict(devPath, devSize, scorer, outPath, doConllPostprocessing);
  }
  
  def runTrain(trainPath: String, trainSize: Int, modelPath: String) {
    checkFileReachableForWrite(modelPath, "modelPath");
    val scorer = runTrain(trainPath, trainSize);
    saveModelFile(scorer, modelPath);
  }
  
  def runTrain(trainPath: String, trainSize: Int): PairwiseScorer = {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val queryCounts: QueryCountsBundle = if (Driver.queryCountsFile != "") {
      QueryCountsBundle.createFromFile(Driver.queryCountsFile) 
    } else {
      Logger.logss("Not loading query file");
      null;
    }
    val trainDocs = loadCorefDocs(trainPath, trainSize, numberGenderComputer, Driver.trainOnGold);
    val trainDocGraphsOrigOrder = trainDocs.map(new DocumentGraph(_, true));
    val trainDocGraphs = if (Driver.randomizeTrain) new scala.util.Random(0).shuffle(trainDocGraphsOrigOrder.sortBy(_.corefDoc.rawDoc.printableDocName)) else trainDocGraphsOrigOrder
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
//    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Seq(Seq(-1), Seq(0), Seq(1)), Seq(Seq(0)), Seq(Seq(-2), Seq(0)), Driver.lexicalFeatCutoff);
    
    Logger.logss("PRUNING BY DISTANCE");
    DocumentGraph.pruneEdgesAll(trainDocGraphs, new PruningStrategy(Driver.pruningStrategy), null);
    
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, Driver.pairwiseFeats, Driver.conjType, lexicalCounts, queryCounts);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
    basicFeaturizer.printFeatureTemplateCounts()

    val basicInferencer = new DocumentInferencerBasic()
    val lossFcnObjFirstPass = PairwiseLossFunctions(Driver.lossFcn);
    val firstPassWeights = featurizerTrainer.train(trainDocGraphs,
                                                   basicFeaturizer,
                                                   Driver.eta,
                                                   Driver.reg,
                                                   lossFcnObjFirstPass,
                                                   Driver.numItrs,
                                                   basicInferencer);
    new PairwiseScorer(basicFeaturizer, firstPassWeights);
  }
  
  def runEvaluate(devPath: String, devSize: Int, modelPath: String) {
    runEvaluate(devPath, devSize, loadModelFile(modelPath));
  }
  
  def runEvaluate(devPath: String, devSize: Int, scorer: PairwiseScorer) {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val devDocs = loadCorefDocs(devPath, devSize, numberGenderComputer, false);
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    DocumentGraph.pruneEdgesAll(devDocGraphs, new PruningStrategy(Driver.pruningStrategy), null);
    new CorefFeaturizerTrainer().featurizeBasic(devDocGraphs, scorer.featurizer);  // dev docs already know they are dev docs so they don't add features
    Logger.startTrack("Decoding dev");
    val basicInferencer = new DocumentInferencerBasic();
    Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, basicInferencer, scorer, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
    Logger.endTrack();
  }
  
  def runPredict(devPath: String, devSize: Int, modelPath: String, outPath: String, doConllPostprocessing: Boolean) {
    runPredict(devPath, devSize, loadModelFile(modelPath), outPath, doConllPostprocessing);
  }
  
  def runPredict(devPath: String, devSize: Int, scorer: PairwiseScorer, outPath: String, doConllPostprocessing: Boolean) {
    checkFileReachableForWrite(outPath, "outputPath");
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val devDocs = loadCorefDocs(devPath, devSize, numberGenderComputer, false);
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    DocumentGraph.pruneEdgesAll(devDocGraphs, new PruningStrategy(Driver.pruningStrategy), null);
    new CorefFeaturizerTrainer().featurizeBasic(devDocGraphs, scorer.featurizer);  // dev docs already know they are dev docs so they don't add features
    Logger.startTrack("Decoding dev");
    val basicInferencer = new DocumentInferencerBasic();
    val (allPredBackptrs, allPredClusterings) = basicInferencer.viterbiDecodeAllFormClusterings(devDocGraphs, scorer);
    val writer = IOUtils.openOutHard(outPath);
    for (i <- 0 until devDocGraphs.size) {
      val outputClustering = new OrderedClusteringBound(devDocGraphs(i).getMentions, allPredClusterings(i));
      ConllDocWriter.writeDoc(writer, devDocGraphs(i).corefDoc.rawDoc, if (doConllPostprocessing) outputClustering.postprocessForConll() else outputClustering);
    }
    writer.close();
  }
  
  def runNewOnlyTwoPass(trainPath: String, trainSize: Int, devPath: String, devSize: Int) {
    checkFileReachableForRead(Driver.conllEvalScriptPath, "conllEvalScriptPath");
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val queryCounts: QueryCountsBundle = (if (Driver.queryCountsFile != "") QueryCountsBundle.createFromFile(Driver.queryCountsFile) else null);
    val trainDocs = loadCorefDocs(trainPath, trainSize, numberGenderComputer, Driver.trainOnGold);
    val devDocs = loadCorefDocs(devPath, devSize, numberGenderComputer, false);
    val trainDocGraphs = trainDocs.map(new DocumentGraph(_, true));
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
//    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Seq(Seq(-1), Seq(0), Seq(1)), Seq(Seq(0)), Seq(Seq(-2), Seq(0)), Driver.lexicalFeatCutoff);
    preprocessDocsClusterInfo(trainDocGraphs ++ devDocGraphs);
    
    val featurizerTrainer = new CorefFeaturizerTrainer();
    val lossFcnObjFirstPass = PairwiseLossFunctions(Driver.lossFcn);
    Logger.logss("PRUNING BY DISTANCE");
    DocumentGraph.pruneEdgesAll(trainDocGraphs, new PruningStrategy(Driver.pruningStrategy), null);
    DocumentGraph.pruneEdgesAll(devDocGraphs, new PruningStrategy(Driver.pruningStrategy), null);

    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, Driver.pairwiseFeats, Driver.conjType, lexicalCounts, queryCounts);
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);  // dev docs already know they are dev docs so they don't add features
    basicFeaturizer.printFeatureTemplateCounts()

    val basicInferencer = if (Driver.inferenceType != InferenceType.BINARY) {
      new DocumentInferencerBasic()
    } else {
      new DocumentInferencerBinary(Driver.binaryLogThreshold * (if (Driver.binaryNegateThreshold) -1.0 else 1.0),
                                   Driver.binaryClusterType,
                                   Driver.binaryNegativeClassWeight);
    }
    val firstPassWeights = featurizerTrainer.train(trainDocGraphs,
                                                   basicFeaturizer,
                                                   Driver.eta,
                                                   Driver.reg,
                                                   lossFcnObjFirstPass,
                                                   Driver.numItrs,
                                                   basicInferencer);
    val firstPassPairwiseScorer = new PairwiseScorer(basicFeaturizer, firstPassWeights);
    Logger.startTrack("Decoding dev");
    Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, basicInferencer, firstPassPairwiseScorer, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
    Logger.endTrack();
    if (Driver.inferenceType == InferenceType.BINARY) {
      return;
    }
    
    Logger.logss("PRUNING WITH ACTUAL STRATEGY");
    Logger.logss("Memory before pruning: " + SysInfoUtils.getUsedMemoryStr());
    DocumentGraph.pruneEdgesAll(trainDocGraphs, new PruningStrategy(Driver.pruningStrategySecondPass), firstPassPairwiseScorer);
    DocumentGraph.pruneEdgesAll(devDocGraphs, new PruningStrategy(Driver.pruningStrategySecondPass), firstPassPairwiseScorer);
    Logger.logss("Memory after pruning: " + SysInfoUtils.getUsedMemoryStr());
      
    val lossFcnObjSecondPass = PairwiseLossFunctions(Driver.lossFcnSecondPass);
    
    // Learn the advanced model
    Logger.logss("Refeaturizing for second pass");
    val secondPassFeatureIndexer = new Indexer[String]();
    secondPassFeatureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val secondPassBasicFeaturizer = new PairwiseIndexingFeaturizerJoint(secondPassFeatureIndexer, Driver.pairwiseFeatsSecondPass, Driver.conjTypeSecondPass, lexicalCounts, queryCounts);
    // Explicitly clear the caches and refeaturize the documents
    trainDocGraphs.foreach(_.cacheEmpty = true);
    featurizerTrainer.featurizeBasic(trainDocGraphs, secondPassBasicFeaturizer);
    devDocGraphs.foreach(_.cacheEmpty = true);
    featurizerTrainer.featurizeBasic(devDocGraphs, secondPassBasicFeaturizer);  // dev docs already know they are dev docs so they don't add features
    Logger.logss(secondPassFeatureIndexer.size() + " features after refeaturization");
    val (secondPassFeaturizer, secondPassInferencer) = if (Driver.inferenceType == InferenceType.LOOPY) {
      val numFeatsBeforeLoopyPass = secondPassBasicFeaturizer.getIndexer.size();
      featurizerTrainer.featurizeLoopyAddToIndexer(trainDocGraphs, secondPassBasicFeaturizer);
      Logger.logss("Features before loopy pass: " + numFeatsBeforeLoopyPass + ", after: " + secondPassBasicFeaturizer.getIndexer.size());
      val inferencer = new DocumentInferencerLoopy();
      (secondPassBasicFeaturizer, inferencer);
    } else if (Driver.inferenceType == InferenceType.RAHMAN) {
      val numFeatsBeforeLoopyPass = secondPassBasicFeaturizer.getIndexer.size();
      val entityFeaturizer = new EntityFeaturizer(Driver.clusterFeats);
      featurizerTrainer.featurizeRahmanAddToIndexer(trainDocGraphs, secondPassBasicFeaturizer, entityFeaturizer);
      Logger.logss("Features before Rahman pass: " + numFeatsBeforeLoopyPass + ", after: " + secondPassBasicFeaturizer.getIndexer.size());
      val inferencer = new DocumentInferencerRahman(entityFeaturizer, secondPassBasicFeaturizer.getIndexer, Driver.rahmanTrainType);
      (secondPassBasicFeaturizer, inferencer);
    } else {
      (secondPassBasicFeaturizer, new DocumentInferencerBasic());
    }
    
    val secondPassWeights = featurizerTrainer.train(trainDocGraphs,
                                                    secondPassFeaturizer,
                                                    Driver.eta,
                                                    Driver.reg,
                                                    lossFcnObjSecondPass,
                                                    Driver.numItrsSecondPass,
                                                    secondPassInferencer);
    
    val secondPassPairwiseScorer = new PairwiseScorer(secondPassFeaturizer, secondPassWeights);
    Logger.startTrack("Decoding dev");
    Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, secondPassInferencer, secondPassPairwiseScorer, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
    Logger.endTrack();  
  }
}