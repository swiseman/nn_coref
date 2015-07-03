package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.Indexer

trait DocumentInferencer {
  
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Double];
  
  def computeLikelihood(docGraph: DocumentGraph,
                        pairwiseScorer: PairwiseScorer,
                        lossFcn: (CorefDoc, Int, Int) => Double): Double;
  
  def addUnregularizedStochasticGradient(docGraph: DocumentGraph,
                                         pairwiseScorer: PairwiseScorer,
                                         lossFcn: (CorefDoc, Int, Int) => Double,
                                         gradient: Array[Double]);
  
  def viterbiDecode(docGraph: DocumentGraph,
                    pairwiseScorer: PairwiseScorer): Array[Int];
  
  def finishPrintStats();
  
  def viterbiDecodeAll(docGraphs: Seq[DocumentGraph], pairwiseScorer: PairwiseScorer): Array[Array[Int]] = {
    val allPredBackptrs = new Array[Array[Int]](docGraphs.size);
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      Logger.logs("Decoding " + i);
      val predBackptrs = viterbiDecode(docGraph, pairwiseScorer);
      allPredBackptrs(i) = predBackptrs;
    }
    allPredBackptrs;
  }
  
  def viterbiDecodeAllFormClusterings(docGraphs: Seq[DocumentGraph], pairwiseScorer: PairwiseScorer): (Array[Array[Int]], Array[OrderedClustering]) = {
    val allPredBackptrs = viterbiDecodeAll(docGraphs, pairwiseScorer);
    val allPredClusteringsSeq = (0 until docGraphs.size).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    (allPredBackptrs, allPredClusteringsSeq.toArray)
  }
}