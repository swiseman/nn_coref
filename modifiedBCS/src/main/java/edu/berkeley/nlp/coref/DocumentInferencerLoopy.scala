package edu.berkeley.nlp.coref
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.coref.bp.DocumentFactorGraph

class DocumentInferencerLoopy extends DocumentInferencer {
  val goldFactorGraphCache = new HashMap[DocumentGraph, DocumentFactorGraph]();
  val guessFactorGraphCache = new HashMap[DocumentGraph, DocumentFactorGraph]();
  
  var renderCounter = 0;
  var wCounter = 0;
  var egCounter = 0;
  
  val uncertaintyStats = ArrayBuffer.fill(6)(0);
  val unkStats = ArrayBuffer.fill(8)(0);
  
  val NumBpIters = 5; //15; // 5
  
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Double] = Array.fill(featureIndexer.size())(0.0);
  
  /**
   * N.B. always returns a reference to the same matrix, so don't call twice in a row and
   * attempt to use the results of both computations
   */
  def instantiateGraph(docGraph: DocumentGraph,
                       featurizer: PairwiseIndexingFeaturizer,
                       gold: Boolean): DocumentFactorGraph = {
    // Construct the factor graph, instantiating if necessary
    val docFactorGraph = if (gold) {
      if (!goldFactorGraphCache.contains(docGraph)) {
        goldFactorGraphCache.put(docGraph, new DocumentFactorGraph(docGraph, featurizer, gold));
      }
      goldFactorGraphCache(docGraph);
    } else {
      if (!guessFactorGraphCache.contains(docGraph)) {
        guessFactorGraphCache.put(docGraph, new DocumentFactorGraph(docGraph, featurizer, gold));
      }
      guessFactorGraphCache(docGraph);
    }
    docFactorGraph;
  }
  
  def computeAndStoreMarginals(docFactorGraph: DocumentFactorGraph,
                               pairwiseScorer: PairwiseScorer,
                               lossFcn: (CorefDoc, Int, Int) => Double) {
//    Logger.logss("Computing marginals, gold = " + docFactorGraph.gold);
    val time = System.nanoTime();
    docFactorGraph.setWeights(pairwiseScorer, lossFcn);
//    Logger.logss(docFactorGraph.renderProperties());
    // Now do inference
    for (i <- 0 until NumBpIters) {
      docFactorGraph.passMessagesOneRound(i == 0 || i == NumBpIters - 1);
//      Logger.logss("LL " + i + ": " + computeLikelihood(docFactorGraph));
//      Logger.logss("MESSAGE PASSING ROUND " + i);
//      Logger.logss(docFactorGraph.renderProperties());
    } 
//    Logger.logss("Inference time: " + (System.nanoTime() - time)/1000000 + " millis on size " +
//                 docFactorGraph.docGraph.size + " with " + docFactorGraph.allFactors.size + " factors, " + docFactorGraph.nodeMillis + " millis on nodes, " + docFactorGraph.factorMillis + " millis on factors");
    // Purely for rendering purposes
  }
  
  def computeLikelihood(docGraph: DocumentGraph,
                        pairwiseScorer: PairwiseScorer,
                        lossFcn: (CorefDoc, Int, Int) => Double): Double = {
    val time = System.nanoTime();
    val docFactorGraph = instantiateGraph(docGraph, pairwiseScorer.featurizer, false);
    computeAndStoreMarginals(docFactorGraph, pairwiseScorer, lossFcn);
    computeLikelihood(docFactorGraph);
  }
  
  private def computeLikelihood(docFactorGraph: DocumentFactorGraph): Double = {
    var likelihood = 0.0;
    val goldAntecedents = docFactorGraph.docGraph.getGoldAntecedentsUnderCurrentPruning();
    for (i <- 0 until docFactorGraph.docGraph.size) {
      val antecedents = goldAntecedents(i);
      val currNodeMarginals = docFactorGraph.getDenseAntecedentNodeMarginals(i);
      var currProb = 0.0;
      for (j <- antecedents) {
        currProb += currNodeMarginals(j);
      }
      var currLogProb = Math.log(currProb);
      if (currLogProb.isInfinite()) {
        currLogProb = -30;
      }
      likelihood += currLogProb;
    }
    likelihood;
  }
  
  def addUnregularizedStochasticGradient(docGraph: DocumentGraph,
                                         pairwiseScorer: PairwiseScorer,
                                         lossFcn: (CorefDoc, Int, Int) => Double,
                                         gradient: Array[Double]) = {
    val predDocFactorGraph = instantiateGraph(docGraph, pairwiseScorer.featurizer, false);
    computeAndStoreMarginals(predDocFactorGraph, pairwiseScorer, lossFcn);
    predDocFactorGraph.addExpectedFeatureCountsToGradient(-1.0, gradient);
    val goldDocFactorGraph = instantiateGraph(docGraph, pairwiseScorer.featurizer, true);
    computeAndStoreMarginals(goldDocFactorGraph, pairwiseScorer, lossFcn);
    goldDocFactorGraph.addExpectedFeatureCountsToGradient(1.0, gradient);
    
//    val renderWeights = wCounter % 50 == 0;
    val renderMarginals = (renderCounter == 50 || renderCounter % 1000 == 999);
    val renderWeights = (wCounter == 50 || wCounter % 1000 == 999);
    
    // RENDERING OF WEIGHTS
    if (renderWeights) {
      val fetchWeight = (featName: String) => {
        val idx = pairwiseScorer.featurizer.getIndexer.indexOf(featName)
        if (idx == -1) 1337.0 else pairwiseScorer.weights(idx);
      }
      if (Driver.clusterFeats.contains("latent")) {
        val pairwiseFeats = pairwiseScorer.featurizer.getIndexer.getObjects.asScala.filter(_.contains("LatentPairwise"));
        for (i <- 0 until Math.min(pairwiseFeats.size, 100)) {
          Logger.logss("Weight of " + pairwiseFeats(i) + ": " + fetchWeight(pairwiseFeats(i)));
        }
        val projFeats = pairwiseScorer.featurizer.getIndexer.getObjects.asScala.filter(_.contains("LatentProj"));
        if (!Driver.clusterFeats.contains("projagree") && !projFeats.isEmpty) {
          val typeStrs = if (Driver.clusterFeats.contains("projfine")) {
            Seq("T0", "T1", "T2");
          } else {
            Seq("")
          }
          for (typeStr <- typeStrs) {
            for (cid <- 0 until docGraph.numClusterers) {
              Logger.logss("Mapping matrix " + typeStr + "C" + cid);
              Logger.logss("\t" + (0 until predDocFactorGraph.numCorefClustersVect(cid)).foldLeft("")((curr, nextI) => curr + nextI + "\t"));
              for (i <- 0 until predDocFactorGraph.numLatentClustersVect(cid)) {
                var strThisLine = i + "\t";
                for (j <- 0 until predDocFactorGraph.numCorefClustersVect(cid)) {
                  strThisLine += GUtil.fmt(fetchWeight("LatentProj" + typeStr + "C" + cid + "-" + i + "-" + j)) + "\t";
                }
                Logger.logss(strThisLine);
              }
            }
          }
        } else {
          projFeats.slice(0, 100).foreach(feat => Logger.logss(feat + ": " + fetchWeight(feat)));
        }
      }
    }
    wCounter += 1;
    
    if (renderMarginals) {
      Logger.logss("PRED");
      Logger.logss(predDocFactorGraph.renderLatentInfo());
      Logger.logss("GOLD");
      Logger.logss(goldDocFactorGraph.renderLatentInfo());
    }
    renderCounter += 1;
    
    // EMPIRICAL GRADIENT CHECK
//    if (egCounter % 50 == 49) {
//      Logger.logss("DocumentInferencerLoopy: empirical gradient check");
//      val backupWeights = new Array[Double](pairwiseScorer.weights.size);
//      Array.copy(pairwiseScorer.weights, 0, backupWeights, 0, backupWeights.size);
//      val backupGradient = Array.fill(backupWeights.size)(0.0);
//      predDocFactorGraph.addExpectedFeatureCountsToGradient(-1.0, backupGradient);
//      goldDocFactorGraph.addExpectedFeatureCountsToGradient(1.0, backupGradient);
//      val llBefore = computeLikelihood(docGraph, new PairwiseScorer(pairwiseScorer.featurizer, backupWeights), lossFcn);
//      for (i <- (0 until 30) ++ (backupWeights.length - 30 until backupWeights.length)) {
//        val delta = 1e-7;
//        backupWeights(i) += delta;
//        val llAfter = computeLikelihood(docGraph, new PairwiseScorer(pairwiseScorer.featurizer, backupWeights), lossFcn);
//        backupWeights(i) -= delta;
//        if (Math.abs(backupGradient(i) - (llAfter - llBefore)/delta) > 1e-6) {
//          Logger.logss("Bump test problem: " + i + ": gradient = " + backupGradient(i) + ", change = " + ((llAfter - llBefore)/delta));
//        }
//      }
//      System.exit(0);
//    }
//    egCounter += 1;
  }
  
  def viterbiDecode(docGraph: DocumentGraph, scorer: PairwiseScorer): Array[Int] = {
    val docFactorGraph = instantiateGraph(docGraph, scorer.featurizer, false);
    computeAndStoreMarginals(docFactorGraph, scorer, PairwiseLossFunctions.noLoss);
//    if (testSet) {
//      val newUncertaintyStats = docFactorGraph.computeUncertaintyStatistics();
//      (0 until uncertaintyStats.size).foreach(i => uncertaintyStats(i) += newUncertaintyStats(i));
//      val newUnkStats = docFactorGraph.computeUnkStatistics();
//      (0 until unkStats.size).foreach(i => unkStats(i) += newUnkStats(i));
//    }
    // Currently MBR decoding, no exping of weights
    val backpointers = new Array[Int](docGraph.size);
    for (i <- 0 until docGraph.size) {
      var bestIdx = -1;
      var bestScore = Double.NegativeInfinity;
      val marginals = docFactorGraph.getDenseAntecedentNodeMarginals(i);
      for (j <- 0 to i) {
        val currScore = marginals(j);
        if (bestIdx == -1 || currScore > bestScore) {
          bestIdx = j;
          bestScore = currScore;
        }
      }
      backpointers(i) = bestIdx;
    }
    backpointers;
  }
  
  def finishPrintStats() = {
    Logger.logss("Uncertainty statistics: " + uncertaintyStats);
    Logger.logss("Unknowns: " + unkStats);
  }
}