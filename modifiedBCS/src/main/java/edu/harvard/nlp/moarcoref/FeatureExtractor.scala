package edu.harvard.nlp.moarcoref;

import java.io.PrintWriter

import scala.collection.JavaConverters.mapAsScalaMapConverter
import scala.collection.immutable.TreeMap

import edu.berkeley.nlp.coref.CorefFeaturizerTrainer
import edu.berkeley.nlp.coref.CorefSystem
import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.coref.NumberGenderComputer
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizerJoint
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.coref.sem.QueryCountsBundle
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger

object FeatureExtractor {
    
  def writeSeparatedFeatsAndOraclePredClustering(smaller:Boolean) {
    var pfx = (if (smaller) "SMALL" else "BIG");
    Logger.logss("Using conjType = " + MiniDriver.conjType);
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(MiniDriver.numberGenderDataPath);
    // require(!MiniDriver.trainOnGold);

    var trainDocs = CorefSystem.loadCorefDocs(MiniDriver.trainPath, MiniDriver.trainSize, numberGenderComputer, MiniDriver.useGoldMentions);
    var trainDocGraphsOrigOrder = trainDocs.map(new DocumentGraph(_, true));
    var trainDocGraphs = if (MiniDriver.randomizeTrain) new scala.util.Random(0).shuffle(trainDocGraphsOrigOrder.sortBy(_.corefDoc.rawDoc.printableDocName)) else trainDocGraphsOrigOrder;   
    
    Logger.logss(trainDocGraphs.size + " many train docs");
    val totalMents = trainDocGraphs.foldLeft(0)((total, curr) => total + curr.size);
    val lexicalCounts = MoarLexicalCountsBundle.countLexicalItems(trainDocs, MiniDriver.lexicalFeatCutoff, MiniDriver.bilexicalFeatCutoff);
    val queryCounts: QueryCountsBundle = null;
    val featurizerTrainer = new CorefFeaturizerTrainer();
 
    // extract anaphoricity features
    var anaphFeatureIndexer = new Indexer[String]();
    anaphFeatureIndexer.getIndex(SeparatingFeaturizer.UnkFeatName);
    // last true parameter to function below means it's in anaphoricity mode
    var anaphFeaturizer = new SmallerSeparatingFeaturizer(anaphFeatureIndexer, MiniDriver.pairwiseFeats, MiniDriver.conjType, lexicalCounts, queryCounts, true); //anaphoricityMode=true    
    featurizerTrainer.featurizeBasic(trainDocGraphs, anaphFeaturizer);
    anaphFeaturizer.printFeatureTemplateCounts();
    // write our features to a file
    TextPickler.writeAnaphFeats(trainDocGraphs, pfx + "-" + MiniDriver.pairwiseFeats + "-" + "anaphTrainFeats.txt");
    
    // write anaph feature mapping
    val printerAnaph = new PrintWriter(pfx+"-"+MiniDriver.pairwiseFeats + "-" + "anaphMapping.txt");
    var invMap = anaphFeatureIndexer.getMap().asScala.map(_.swap); // asScala is magic
    var tmap = TreeMap(invMap.toSeq:_*); // sort the map
    for ((idx,str) <- tmap){ 
      printerAnaph.println(idx + " : " + str);
    }
    printerAnaph.flush();
    printerAnaph.close();
    
    // write oracle pred clustering for train
    TextPickler.writePredOracleClusterings(trainDocGraphs, pfx+"TrainOPCs.txt");
    
    // now do pairwise features
    trainDocGraphsOrigOrder = trainDocs.map(new DocumentGraph(_, true));
    trainDocGraphs = if (MiniDriver.randomizeTrain) new scala.util.Random(0).shuffle(trainDocGraphsOrigOrder.sortBy(_.corefDoc.rawDoc.printableDocName)) else trainDocGraphsOrigOrder;
    var pwFeatureIndexer = new Indexer[String]();
    pwFeatureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    // below we set anaphoricityMode = false
    var pwFeaturizer:PairwiseIndexingFeaturizer = null;
    if (smaller){
      pwFeaturizer = new SmallerSeparatingFeaturizer(pwFeatureIndexer, MiniDriver.pairwiseFeats, MiniDriver.conjType, lexicalCounts, queryCounts, false); //anaphoricityMode=false
    } else{
      pwFeaturizer = new SeparatingFeaturizer(pwFeatureIndexer, MiniDriver.pairwiseFeats, MiniDriver.conjType, lexicalCounts, queryCounts, false); //anaphoricityMode=false
    }
    //var pwFeaturizer = new SeparatingFeaturizer(pwFeatureIndexer, MiniDriver.pairwiseFeats, MiniDriver.conjType, lexicalCounts, queryCounts, false); //anaphoricityMode=false
    featurizerTrainer.featurizeBasic(trainDocGraphs, pwFeaturizer);
    pwFeaturizer.printFeatureTemplateCounts;
    // write pairwise train features
    TextPickler.writePWFeats(trainDocGraphs, pwFeatureIndexer.size(), pfx + "-" + MiniDriver.pairwiseFeats + "-" + "pwTrainFeats.txt");

    // write pw feature mapping
    val printerPW = new PrintWriter(pfx+"-"+ MiniDriver.pairwiseFeats + "-" + "pwMapping.txt");
    invMap = pwFeatureIndexer.getMap().asScala.map(_.swap); // asScala is magic
    tmap = TreeMap(invMap.toSeq:_*); // sort the map
    for ((idx,str) <- tmap){
      printerPW.println(idx + " : " + str);
    }
    printerPW.flush();
    printerPW.close();    
    
    // hopefully helps with gc
    trainDocs = null;
    trainDocGraphsOrigOrder = null;
    trainDocGraphs = null;

    var devDocs = CorefSystem.loadCorefDocs(MiniDriver.devPath, MiniDriver.devSize, numberGenderComputer, MiniDriver.useGoldMentions);
    var devDocGraphs = devDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    featurizerTrainer.featurizeBasic(devDocGraphs, anaphFeaturizer); // dev docs already know they are dev docs so they don't add features
    TextPickler.writeAnaphFeats(devDocGraphs, pfx + "-" + MiniDriver.pairwiseFeats + "-" + "anaphDevFeats.txt");
    devDocGraphs = devDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    featurizerTrainer.featurizeBasic(devDocGraphs,pwFeaturizer);
    TextPickler.writePWFeats(devDocGraphs, pwFeatureIndexer.size(), pfx + "-" +  MiniDriver.pairwiseFeats + "-" + "pwDevFeats.txt");
    
    // write dev oracle predicted clustering
    TextPickler.writePredOracleClusterings(devDocGraphs, pfx+"DevOPCs.txt"); 
       
    // do test docs
    devDocs = null;
    devDocGraphs = null;
    var testDocs = CorefSystem.loadCorefDocs(MiniDriver.testPath, MiniDriver.testSize, numberGenderComputer, MiniDriver.useGoldMentions);
      
    var testDocGraphs = testDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    featurizerTrainer.featurizeBasic(testDocGraphs, anaphFeaturizer); // test docs already know they are test docs so they don't add features
    TextPickler.writeAnaphFeats(testDocGraphs, pfx + "-" +  MiniDriver.pairwiseFeats + "-" + "anaphTestFeats.txt");
    testDocGraphs = testDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    featurizerTrainer.featurizeBasic(testDocGraphs,pwFeaturizer);
    TextPickler.writePWFeats(testDocGraphs, pwFeatureIndexer.size(), pfx + "-" + MiniDriver.pairwiseFeats + "-" + "pwTestFeats.txt");  
  } 
}
