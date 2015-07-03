package edu.berkeley.nlp.coref
import scala.collection.mutable.HashMap

import edu.berkeley.nlp.futile.fig.basic.Indexer

// TODO: Tune both of these, also try out some subsampling/reweighting approaches
class DocumentInferencerBinary(val logThreshold: Double,
                               val clusterType: String,
                               val negativeClassWeight: Double) extends DocumentInferencer {
  
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Double] = Array.fill(featureIndexer.size())(0.0);
  
  private def subsample(docGraph: DocumentGraph, i: Int): Seq[Int] = {
    (0 until i);
  }
  
  def computeLikelihood(docGraph: DocumentGraph,
                        pairwiseScorer: PairwiseScorer,
                        lossFcn: (CorefDoc, Int, Int) => Double): Double = {
    val (featsChart, scoresChart) = docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer);
    var likelihood = 0.0;
    for (i <- 0 until docGraph.size) {
      for (j <- subsample(docGraph, i)) {
        val pos = docGraph.isGoldNoPruning(i, j); 
        var increment = if (pos) {
          scoresChart(i)(j) - Math.log(1 + Math.exp(scoresChart(i)(j)))
        } else {
          negativeClassWeight * -Math.log(1 + Math.exp(scoresChart(i)(j)));
        }
        if (increment.isNegInfinity) {
          increment = -30;
        }
        likelihood += increment;
      }
    }
    likelihood;
  }
  
  def addUnregularizedStochasticGradient(docGraph: DocumentGraph,
                                         pairwiseScorer: PairwiseScorer,
                                         lossFcn: (CorefDoc, Int, Int) => Double,
                                         gradient: Array[Double]) = {
    val (featsChart, scoresChart) = docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer);
    for (i <- 0 until docGraph.size) {
      for (j <- subsample(docGraph, i)) {
        val expedScore = Math.exp(scoresChart(i)(j));
        if (docGraph.isGoldNoPruning(i, j)) {
          addToGradient(featsChart(i)(j), 1.0 - expedScore/(1.0 + expedScore), gradient);
        } else {
          addToGradient(featsChart(i)(j), negativeClassWeight * -expedScore/(1.0 + expedScore), gradient);
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
    clusterType match {
      case "CLOSEST_FIRST" => {
        (0 until docGraph.size).map(i => {
          var nearest = i;
          for (j <- i-1 to 0 by -1) {
            if (nearest == i && scoresChart(i)(j) > logThreshold) {
              nearest = j;
            }
          }
          nearest;
        }).toArray;
      }
      case "BEST_FIRST" => {
        (0 until docGraph.size).map(i => {
          var best = i;
          var bestScore = Double.NegativeInfinity;
          for (j <- i-1 to 0 by -1) {
            if (scoresChart(i)(j) > logThreshold && scoresChart(i)(j) > bestScore) {
              best = j;
              bestScore = scoresChart(i)(j);
            }
          }
          best;
        }).toArray;
      }
      case _ => { // TRANSITIVE_CLOSURE
        var mapping = new HashMap[Int,Int]();
        var nextClusterIndex = 0;
        for (i <- 0 until docGraph.size) {
          var edgeAlreadyFound = false;
          for (j <- 0 until i) {
            if (scoresChart(i)(j) > logThreshold) {
              var antecedentCluster = mapping(j);
              // Merge the two
              if (edgeAlreadyFound && antecedentCluster != mapping(i)) {
                var newCluster = mapping(i);
                for (mentIdx <- mapping.keySet) {
                  if (mapping(mentIdx) == antecedentCluster) {
                    mapping(mentIdx) = newCluster;
                  }
                }
              } else {
                edgeAlreadyFound = true;
                mapping(i) = antecedentCluster;
              }
            }
          }
          if (!edgeAlreadyFound) {
            mapping(i) = nextClusterIndex;
            nextClusterIndex += 1;
          }
        }
        (0 until docGraph.size).map(i => {
          var backptr = i;
          for (j <- 0 until i) {
            if (mapping(j) == mapping(i)) {
              backptr = j;
            }
          }
          backptr;
        }).toArray;
      }
    }
  }
  
  def finishPrintStats() = {}
}