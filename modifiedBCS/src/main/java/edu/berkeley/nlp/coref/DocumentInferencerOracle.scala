package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.fig.basic.Indexer

class DocumentInferencerOracle extends DocumentInferencer {
  
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Double] = Array.fill(featureIndexer.size())(0.0);
  
  def computeLikelihood(docGraph: DocumentGraph,
                        pairwiseScorer: PairwiseScorer,
                        lossFcn: (CorefDoc, Int, Int) => Double) = {
    0.0;
  }
  
  def addUnregularizedStochasticGradient(docGraph: DocumentGraph,
                                         pairwiseScorer: PairwiseScorer,
                                         lossFcn: (CorefDoc, Int, Int) => Double,
                                         gradient: Array[Double]) = {
  }
  
  def viterbiDecode(docGraph: DocumentGraph,
                    pairwiseScorer: PairwiseScorer): Array[Int] = {
    val clustering = docGraph.getOraclePredClustering();
    val resultSeq = for (i <- 0 until docGraph.size) yield {
      val immediateAntecedentOrMinus1 = clustering.getImmediateAntecedent(i);
      if (immediateAntecedentOrMinus1 == -1) {
        i;
      } else {
        docGraph.getMentions.indexOf(immediateAntecedentOrMinus1);
      }
    }
    resultSeq.toArray;
  }
  
  def finishPrintStats() = {}
}