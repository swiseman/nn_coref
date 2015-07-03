package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.fig.basic.Indexer

class DocumentInferencerBasic extends DocumentInferencer {
  
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Double] = Array.fill(featureIndexer.size())(0.0);
  
  /**
   * N.B. always returns a reference to the same matrix, so don't call twice in a row and
   * attempt to use the results of both computations
   */
  private def computeMarginals(docGraph: DocumentGraph,
                               gold: Boolean,
                               lossFcn: (CorefDoc, Int, Int) => Double,
                               pairwiseScorer: PairwiseScorer): Array[Array[Double]] = {
    computeMarginals(docGraph, gold, lossFcn, docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer)._2)
  }
  
  private def computeMarginals(docGraph: DocumentGraph,
                               gold: Boolean,
                               lossFcn: (CorefDoc, Int, Int) => Double,
                               scoresChart: Array[Array[Double]]): Array[Array[Double]] = {
//    var marginals = new Array[Array[Double]](docGraph.doc.predMentions.size());
//    for (i <- 0 until marginals.size) {
//      marginals(i) = Array.fill(i+1)(Double.NegativeInfinity);
//    }
    val marginals = docGraph.cachedMarginalMatrix;
    for (i <- 0 until docGraph.size) {
      var normalizer = 0.0;
      // Restrict to gold antecedents if we're doing gold, but don't load the gold antecedents
      // if we're not.
      val goldAntecedents: Seq[Int] = if (gold) docGraph.getGoldAntecedentsUnderCurrentPruning(i) else null;
      for (j <- 0 to i) {
        // If this is a legal antecedent
        if (!docGraph.isPruned(i, j) && (!gold || goldAntecedents.contains(j))) {
          // N.B. Including lossFcn is okay even for gold because it should be zero
          val unnormalizedProb = Math.exp(scoresChart(i)(j) + lossFcn(docGraph.corefDoc, i, j));
          marginals(i)(j) = unnormalizedProb;
          normalizer += unnormalizedProb;
        } else {
          marginals(i)(j) = 0.0;
        }
      }
      for (j <- 0 to i) {
        marginals(i)(j) /= normalizer;
      }
    }
    marginals;
  }
  
  def computeLikelihood(docGraph: DocumentGraph,
                        pairwiseScorer: PairwiseScorer,
                        lossFcn: (CorefDoc, Int, Int) => Double): Double = {
    var likelihood = 0.0;
    val marginals = computeMarginals(docGraph, false, lossFcn, pairwiseScorer);
    for (i <- 0 until docGraph.size) {
      val goldAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(i);
      var currProb = 0.0;
      for (j <- goldAntecedents) {
        currProb += marginals(i)(j);
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
    val (featsChart, scoresChart) = docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer);
    // N.B. Can't have pred marginals and gold marginals around at the same time because
    // they both live in the same cached matrix
    val predMarginals = this.computeMarginals(docGraph, false, lossFcn, scoresChart);
    for (i <- 0 until docGraph.size) {
      for (j <- 0 to i) {
        if (predMarginals(i)(j) > 1e-20) {
          addToGradient(featsChart(i)(j), -predMarginals(i)(j), gradient);
        }
      }
    }
    val goldMarginals = this.computeMarginals(docGraph, true, lossFcn, scoresChart);
    for (i <- 0 until docGraph.size) {
      for (j <- 0 to i) {
        if (goldMarginals(i)(j) > 1e-20) {
          addToGradient(featsChart(i)(j), goldMarginals(i)(j), gradient);
        }
      }
    }
  }
  
  private def addToGradient(feats: Seq[Int], scale: Double, gradient: Array[Double]) {
    var i = 0;
    while (i < feats.size) {
      val feat = feats(i);
      gradient(feat) += 1.0 * scale;
      i += 1;
    }
  }

  def viterbiDecode(docGraph: DocumentGraph, scorer: PairwiseScorer): Array[Int] = {
    val (featsChart, scoresChart) = docGraph.featurizeIndexAndScoreNonPrunedUseCache(scorer);
    if (Driver.decodeType == "sum") { 
      val backptrs = Decoder.decodeLeftToRightMarginalize(docGraph, (idx: Int) => {
        val probs = scoresChart(idx);
        GUtil.expAndNormalizeiHard(probs);
        probs;
      });
      backptrs;
    } else {
      val backptrs = Decoder.decodeMax(docGraph, (idx: Int) => {
        val probs = scoresChart(idx);
        GUtil.expAndNormalizeiHard(probs);
        probs;
      });
      backptrs;
    }
  }
  
  def finishPrintStats() = {}
}