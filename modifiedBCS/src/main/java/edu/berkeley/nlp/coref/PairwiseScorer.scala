package edu.berkeley.nlp.coref

@SerialVersionUID(1L)
class PairwiseScorer(val featurizer: PairwiseIndexingFeaturizer, val weights: Array[Double]) extends Serializable {
  
  def numWeights = weights.size
  
  def scoreIndexedFeats(feats: Seq[Int]): Double = {
    var featIdx = 0;
    var featTotal = 0.0;
    while (featIdx < feats.size) {
      featTotal += weights(feats(featIdx));
      featIdx += 1;
    }
    featTotal;
  }
}