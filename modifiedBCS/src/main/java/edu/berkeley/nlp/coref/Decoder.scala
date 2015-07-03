package edu.berkeley.nlp.coref
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger

object Decoder {

  def decodeMax(docGraph: DocumentGraph, probFcn: Int => Array[Double]): Array[Int] = {
    val backpointers = new Array[Int](docGraph.size);
    for (i <- 0 until docGraph.size) {
      val allProbs = probFcn(i);
      var bestIdx = -1;
      var bestProb = Double.NegativeInfinity;
      for (j <- 0 to i) {
        val currProb = allProbs(j);
        if (bestIdx == -1 || currProb > bestProb) {
          bestIdx = j;
          bestProb = currProb;
        }
      }
      backpointers(i) = bestIdx;
    }
    backpointers;
  }
  
  def decodeLeftToRightMarginalize(docGraph: DocumentGraph, probFcn: Int => Array[Double]): Array[Int] = {
    val clustersSoFar = new ArrayBuffer[ArrayBuffer[Int]]();
    val backpointers = new Array[Int](docGraph.size);
    for (i <- 0 until docGraph.size) {
      val allProbs = probFcn(i);
      val clusterProbs = clustersSoFar.map(_.foldLeft(0.0)((total, mentIdx) => total + allProbs(mentIdx)));
//      Logger.logss("All probs: " + allProbs.toSeq.zipWithIndex);
//      Logger.logss("Clusters so far: " + clustersSoFar);
//      Logger.logss("Cluster probs: " + clusterProbs.zipWithIndex);
      // Just a sanity-check, should return the same clusters as the max method
//      val clusterProbs = clustersSoFar.map(_.foldLeft(0.0)((total, mentIdx) => Math.max(total, allProbs(mentIdx))));
      val startNewProb = allProbs(i);
      val bestClusterProbAndIdx = clusterProbs.zipWithIndex.foldLeft((0.0, -1))((bestProbAndIdx, currProbAndIdx) => if (bestProbAndIdx._1 < currProbAndIdx._1) currProbAndIdx else bestProbAndIdx);
      if (startNewProb > bestClusterProbAndIdx._1) {
        backpointers(i) = i;
        clustersSoFar += ArrayBuffer(i);
      } else {
        backpointers(i) = clustersSoFar(bestClusterProbAndIdx._2).last;
        clustersSoFar(bestClusterProbAndIdx._2) += i;
      }
    }
    backpointers;
  }
}