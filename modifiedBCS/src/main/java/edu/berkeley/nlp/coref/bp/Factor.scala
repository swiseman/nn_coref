package edu.berkeley.nlp.coref.bp
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.coref.GUtil

trait Factor {
  
  def setWeights(newWeights: Array[Double]);
  
  def receiveMessage(node: Node[_], message: Array[Double]);
  
  def sendMessages();
  
  def getAllAssociatedFeatures(): Array[String];
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]);
}

class AgreementFactor(val nodeOne: Node[String],
                      val nodeTwo: Node[String],
                      val featureMatrix: Array[Array[Seq[String]]],
                      val indexedFeatureMatrix: Array[Array[Seq[Int]]],
                      val defaultValMatrix: Array[Array[Double]]) extends Factor {
  var cachedWeights: Array[Double] = null;
  nodeOne.registerFactor(this);
  nodeTwo.registerFactor(this);
  
  var receivedNodeOneMessage: Array[Double] = null;
  var receivedNodeTwoMessage: Array[Double] = null;
  var sentNodeOneMessage: Array[Double] = new Array[Double](nodeOne.domain.size);
  var sentNodeTwoMessage: Array[Double] = new Array[Double](nodeTwo.domain.size);
  
  def setWeights(newWeights: Array[Double]) {
    this.cachedWeights = newWeights;
    for (i <- 0 until sentNodeOneMessage.length) {
      sentNodeOneMessage(i) = 0;
    }
    for (i <- 0 until sentNodeTwoMessage.length) {
      sentNodeTwoMessage(i) = 0;
    }
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    require(!GUtil.containsNaN(message));
    require(message.size == node.domain.size);
    if (node == nodeOne) {
      receivedNodeOneMessage = message;
    } else if (node == nodeTwo) {
      receivedNodeTwoMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  def factorValue(nodeOneValueIdx: Int, nodeTwoValueIdx: Int): Double = {
    var featValue = 1.0;
    featValue *= Math.exp(defaultValMatrix(nodeOneValueIdx)(nodeTwoValueIdx));
    var featIdx = 0;
    while (featIdx < indexedFeatureMatrix(nodeOneValueIdx)(nodeTwoValueIdx).size) {
      featValue *= Math.exp(cachedWeights(indexedFeatureMatrix(nodeOneValueIdx)(nodeTwoValueIdx)(featIdx)));
      featIdx += 1;
    }
    featValue;
  }
  
  def sendMessages() {
    for (i <- 0 until sentNodeOneMessage.length) {
      sentNodeOneMessage(i) = 0;
    }
    for (i <- 0 until sentNodeTwoMessage.length) {
      sentNodeTwoMessage(i) = 0;
    }
    
    for (i <- 0 until nodeOne.domain.size) {
      // While loop for the inner loop here
      var j = 0;
      while (j < nodeTwo.domain.size) {
        val currFactorValue = factorValue(i, j);
        sentNodeOneMessage(i) += currFactorValue * receivedNodeTwoMessage(j);
        sentNodeTwoMessage(j) += currFactorValue * receivedNodeOneMessage(i);
        j += 1;
      }
    }
    GUtil.normalizeiHard(sentNodeOneMessage);
    GUtil.normalizeiHard(sentNodeTwoMessage);
    
    require(!sentNodeOneMessage.contains(0.0));
    require(!sentNodeTwoMessage.contains(0.0));
    nodeOne.receiveMessage(this, sentNodeOneMessage);
    nodeTwo.receiveMessage(this, sentNodeTwoMessage);
  }
  
  def getAllAssociatedFeatures(): Array[String] = {
    // Flatten matrix of lists of features
    featureMatrix.flatMap(_.flatMap((currFeats: Seq[String]) => currFeats)).toSet.toArray;
  }
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {
    var normalizer = 0.0;
    for (i <- 0 until nodeOne.domain.size) {
      for (j <- 0 until nodeTwo.domain.size) {
        val value = factorValue(i, j) * receivedNodeOneMessage(i) * receivedNodeTwoMessage(j);
        normalizer += value;
      }
    }
    for (i <- 0 until nodeOne.domain.size) {
      for (j <- 0 until nodeTwo.domain.size) {
        val value = factorValue(i, j) * receivedNodeOneMessage(i) * receivedNodeTwoMessage(j);
        var featIdx = 0;
        while (featIdx < indexedFeatureMatrix(i)(j).size) {
          gradient(indexedFeatureMatrix(i)(j)(featIdx)) += scale * value/normalizer;
          featIdx += 1;
        }
      }
    }
  }
}
                      

class PropertyFactor(val selectedAntecedentMentionIdx: Int,
                     val propertyNode: Node[String],
                     val antecedentNode: Node[Int],
                     val antecedentPropertyNode: Node[String],
                     val featureMatrix: Array[Array[Seq[String]]],
                     val indexedFeatureMatrix: Array[Array[Seq[Int]]],
                     val defaultValMatrix: Array[Array[Double]]) extends Factor {
  var cachedWeights: Array[Double] = null;
  require(antecedentPropertyNode.domain == propertyNode.domain);
  propertyNode.registerFactor(this);
  antecedentNode.registerFactor(this);
  antecedentPropertyNode.registerFactor(this);
  var selectedAntecedentValueIdx = -1;
  for (i <- 0 until antecedentNode.domain.size) {
    if (antecedentNode.domain.value(i) == selectedAntecedentMentionIdx) {
      selectedAntecedentValueIdx = i;
    }
  }
  
  var receivedPropertyMessage: Array[Double] = null;
  var receivedAntecedentMessage: Array[Double] = null;
  var receivedAntecedentPropertyMessage: Array[Double] = null;
  var sentPropertyMessage: Array[Double] = new Array[Double](propertyNode.domain.size);
  var sentAntecedentMessage: Array[Double] = new Array[Double](antecedentNode.domain.size);
  var sentAntecedentPropertyMessage: Array[Double] = new Array[Double](antecedentPropertyNode.domain.size);
  
  def setWeights(newWeights: Array[Double]) {
    this.cachedWeights = newWeights;
    for (i <- 0 until sentPropertyMessage.length) {
      sentPropertyMessage(i) = 0;
    }
    for (i <- 0 until sentAntecedentMessage.length) {
      sentAntecedentMessage(i) = 0;
    }
    for (i <- 0 until sentAntecedentPropertyMessage.length) {
      sentAntecedentPropertyMessage(i) = 0;
    }
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    require(!GUtil.containsNaN(message));
    require(message.size == node.domain.size);
    if (node == propertyNode) {
      receivedPropertyMessage = message;
    } else if (node == antecedentNode) {
      receivedAntecedentMessage = message;
    } else if (node == antecedentPropertyNode) {
      receivedAntecedentPropertyMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  def factorValue(propertyValueIdx: Int, antecedentValueIdx: Int, antecedentPropertyValueIdx: Int): Double = {
    if (antecedentValueIdx == selectedAntecedentValueIdx) {
      var featValue = 1.0;
      featValue *= Math.exp(defaultValMatrix(propertyValueIdx)(antecedentPropertyValueIdx));
      var featIdx = 0;
      while (featIdx < indexedFeatureMatrix(propertyValueIdx)(antecedentPropertyValueIdx).size) {
        featValue *= Math.exp(cachedWeights(indexedFeatureMatrix(propertyValueIdx)(antecedentPropertyValueIdx)(featIdx)));
        featIdx += 1;
      }
      featValue;
    } else 1.0;
  }
  
  // TODO: Optimize this if it's slow
  def sendMessages() {
    for (i <- 0 until propertyNode.domain.size) {
      sentPropertyMessage(i) = 0;
    }
    for (i <- 0 until antecedentNode.domain.size) {
      sentAntecedentMessage(i) = 0;
    }
    for (i <- 0 until antecedentPropertyNode.domain.size) {
      sentAntecedentPropertyMessage(i) = 0;
    }
    
//    // Antecedent message
//    var propertyMessageSumForIrrelevantAntecedents = 0.0;
//    for (i <- 0 until propertyNode.domain.size) {
//      for (j <- 0 until antecedentPropertyNode.domain.size) {
//        propertyMessageSumForIrrelevantAntecedents += receivedPropertyMessage(i) * receivedAntecedentPropertyMessage(j);
//      }
//    }
//    for (k <- 0 until antecedentNode.domain.size) {
//      if (k != selectedAntecedentValueIdx) {
//        sentAntecedentMessage(k) = propertyMessageSumForIrrelevantAntecedents;
//      } else {
//        var propertyMessageSumForRelevantAntecedent = 0.0;
//        for (i <- 0 until propertyNode.domain.size) {
//          for (j <- 0 until antecedentPropertyNode.domain.size) {
//            propertyMessageSumForRelevantAntecedent += receivedPropertyMessage(i) * receivedAntecedentPropertyMessage(j) * factorValue(i, k, j);
//          }
//        }
//        sentAntecedentMessage(k) = propertyMessageSumForRelevantAntecedent;
//      }
//    }
//    // Property messages
//    var constantPropertyComponent = 0.0;
//    for (j <- 0 until antecedentPropertyNode.domain.size) {
//      for (k <- 0 until antecedentNode.domain.size) {
//        if (k != selectedAntecedentValueIdx) {
//          constantPropertyComponent += receivedAntecedentPropertyMessage(j) * receivedAntecedentMessage(k);
//        }
//      }
//    }
//    for (i <- 0 until propertyNode.domain.size) {
//      var messageVal = constantPropertyComponent;
//      for (j <- 0 until antecedentPropertyNode.domain.size) {
//        messageVal += receivedAntecedentPropertyMessage(j) * receivedAntecedentMessage(selectedAntecedentValueIdx) * factorValue(i, selectedAntecedentValueIdx, j);
//      }
//      sentPropertyMessage(i) = messageVal;
//    }
//    // Analogous for the other property message
//    
//    var constantAntecedentPropertyComponent = 0.0;
//    for (i <- 0 until propertyNode.domain.size) {
//      for (k <- 0 until antecedentNode.domain.size) {
//        if (k != selectedAntecedentValueIdx) {
//          constantAntecedentPropertyComponent += receivedPropertyMessage(i) * receivedAntecedentMessage(k);
//        }
//      }
//    }
//    for (j <- 0 until antecedentPropertyNode.domain.size) {
//      var messageVal = constantAntecedentPropertyComponent;
//      for (i <- 0 until propertyNode.domain.size) {
//        messageVal += receivedPropertyMessage(i) * receivedAntecedentMessage(selectedAntecedentValueIdx) * factorValue(i, selectedAntecedentValueIdx, j);
//      }
//      sentAntecedentPropertyMessage(j) = messageVal;
//    }
    
    
    // OLD METHOD
    for (i <- 0 until antecedentNode.domain.size) {
      for (j <- 0 until propertyNode.domain.size) {
        // While loop for the inner loop here
        var k = 0;
        while (k < antecedentPropertyNode.domain.size) {
          val currFactorValue = factorValue(j, i, k);
          sentPropertyMessage(j) += currFactorValue * receivedAntecedentMessage(i) * receivedAntecedentPropertyMessage(k);
          sentAntecedentMessage(i) += currFactorValue * receivedPropertyMessage(j) * receivedAntecedentPropertyMessage(k);
          sentAntecedentPropertyMessage(k) += currFactorValue * receivedPropertyMessage(j) * receivedAntecedentMessage(i);
          k += 1;
        }
      }
    }
    GUtil.normalizeiHard(sentPropertyMessage);
    GUtil.normalizeiHard(sentAntecedentMessage);
    GUtil.normalizeiHard(sentAntecedentPropertyMessage);
    
    if (sentPropertyMessage.contains(0.0)) {
      Logger.logss("Received prop message: " + receivedPropertyMessage.toSeq);
      Logger.logss("Received antecedent prop message: " + receivedAntecedentPropertyMessage.toSeq);
      Logger.logss("Received antecedent message: " + receivedAntecedentMessage.toSeq);
      for (i <- 0 until antecedentNode.domain.size) {
        for (j <- 0 until propertyNode.domain.size) {
          // While loop for the inner loop here
          var k = 0;
          while (k < antecedentPropertyNode.domain.size) {
            Logger.logss("Factor value for " + j + " " + i + " " + k + ": " + factorValue(j, i, k));
            k += 1;
          }
        }
      }
      require(false);
    }
    require(!sentAntecedentMessage.contains(0.0));
    require(!sentAntecedentPropertyMessage.contains(0.0));
    propertyNode.receiveMessage(this, sentPropertyMessage);
    antecedentNode.receiveMessage(this, sentAntecedentMessage);
    antecedentPropertyNode.receiveMessage(this, sentAntecedentPropertyMessage);
  }
  
  def getAllAssociatedFeatures(): Array[String] = {
    // Flatten matrix of lists of features
    featureMatrix.flatMap(_.flatMap((currFeats: Seq[String]) => currFeats)).toSet.toArray;
  }
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {
    var normalizer = 0.0;
    for (i <- 0 until antecedentNode.domain.size) {
      for (j <- 0 until propertyNode.domain.size) {
        for (k <- 0 until antecedentPropertyNode.domain.size) {
          val value = factorValue(j, i, k) * receivedPropertyMessage(j) * receivedAntecedentMessage(i) * receivedAntecedentPropertyMessage(k);
          normalizer += value;
        }
      }
    }
    for (i <- 0 until antecedentNode.domain.size) {
      if (antecedentNode.domain.entries(i) == selectedAntecedentMentionIdx) {
        for (j <- 0 until propertyNode.domain.size) {
          for (k <- 0 until antecedentPropertyNode.domain.size) {
            val value = factorValue(j, i, k) * receivedPropertyMessage(j) * receivedAntecedentMessage(i) * receivedAntecedentPropertyMessage(k);
            var featIdx = 0;
            while (featIdx < indexedFeatureMatrix(j)(k).size) {
              gradient(indexedFeatureMatrix(j)(k)(featIdx)) += scale * value/normalizer;
              featIdx += 1;
            }
          }
        }
      }
    }
  }
}

class HardPropertyFactor(val selectedAntecedentMentionIdx: Int,
                         val propertyNode: Node[String],
                         val antecedentNode: Node[Int],
                         val antecedentPropertyNode: Node[String]) extends Factor {
  require(antecedentPropertyNode.domain == propertyNode.domain);
  propertyNode.registerFactor(this);
  antecedentNode.registerFactor(this);
  antecedentPropertyNode.registerFactor(this);
  var selectedAntecedentValueIdx = -1;
  for (i <- 0 until antecedentNode.domain.size) {
    if (antecedentNode.domain.value(i) == selectedAntecedentMentionIdx) {
      selectedAntecedentValueIdx = i;
    }
  }
  
  var receivedPropertyMessage: Array[Double] = null;
  var receivedAntecedentMessage: Array[Double] = null;
  var receivedAntecedentPropertyMessage: Array[Double] = null;
  var sentPropertyMessage: Array[Double] = new Array[Double](propertyNode.domain.size);
  var sentAntecedentMessage: Array[Double] = new Array[Double](antecedentNode.domain.size);
  var sentAntecedentPropertyMessage: Array[Double] = new Array[Double](antecedentPropertyNode.domain.size);
  
  def setWeights(newWeights: Array[Double]) {
    clearMessages();
  }
  
  def clearMessages() {
    var i = 0;
    while (i < propertyNode.domain.size) {
      sentPropertyMessage(i) = 0;
      i += 1;
    }    
    i = 0;
    while (i < propertyNode.domain.size) {
      sentAntecedentPropertyMessage(i) = 0;
      i += 1;
    }
    i = 0;
    while (i < sentAntecedentMessage.length) {
      sentAntecedentMessage(i) = 0;
      i += 1;
    }
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    require(!GUtil.containsNaN(message));
    require(message.size == node.domain.size);
    if (node == propertyNode) {
      receivedPropertyMessage = message;
    } else if (node == antecedentNode) {
      receivedAntecedentMessage = message;
    } else if (node == antecedentPropertyNode) {
      receivedAntecedentPropertyMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  def factorValue(propertyValueIdx: Int, antecedentValueIdx: Int, antecedentPropertyValueIdx: Int): Double = {
    if (antecedentValueIdx == selectedAntecedentValueIdx && propertyValueIdx != antecedentPropertyValueIdx) 0.0 else 1.0;
  }
  
  // TODO: Optimize this if it's slow
  def sendMessages() {
    clearMessages();
    
    // NEW COMPUTATION METHOD
    var i = 0;
    var j = 0;
    var k = 0;
    var propNodeSum = 0.0;
    var antPropNodeSum = 0.0;
    var innerProduct = 0.0;
    i = 0;
    while (i < propertyNode.domain.size) {
      propNodeSum += receivedPropertyMessage(i);
      antPropNodeSum += receivedAntecedentPropertyMessage(i);
      innerProduct += receivedPropertyMessage(i) * receivedAntecedentPropertyMessage(i);
      i += 1;
    }
    var nonselectedAntecedentSum = 0.0;
    var selectedAntecedentVal = 0.0;
    k = 0;
    while (k < antecedentNode.domain.size) {
      if (k != selectedAntecedentValueIdx) {
        nonselectedAntecedentSum += receivedAntecedentMessage(k);
      } else {
        selectedAntecedentVal = receivedAntecedentMessage(k);
      }
      k += 1;
    }
    
    i = 0;
    while (i < propertyNode.domain.size) {
      sentPropertyMessage(i) = antPropNodeSum * nonselectedAntecedentSum + receivedAntecedentPropertyMessage(i) * selectedAntecedentVal;
      i += 1;
    }
    j = 0;
    while (j < antecedentPropertyNode.domain.size) {
      sentAntecedentPropertyMessage(j) = propNodeSum * nonselectedAntecedentSum + receivedPropertyMessage(j) * selectedAntecedentVal;
      j += 1;
    }
    k = 0;
    while (k < antecedentNode.domain.size) {
      if (k == selectedAntecedentValueIdx) {
        sentAntecedentMessage(k) = innerProduct;
      } else {
        sentAntecedentMessage(k) = propNodeSum * antPropNodeSum;
      }
      k += 1;
    }
    
    // OLD METHOD
//    for (i <- 0 until antecedentNode.domain.size) {
//      for (j <- 0 until propertyNode.domain.size) {
//        var k = 0;
//        while (k < antecedentPropertyNode.domain.size) {
//          val currFactorValue = factorValue(j, i, k);
//          sentPropertyMessage(j) += currFactorValue * receivedAntecedentMessage(i) * receivedAntecedentPropertyMessage(k);
//          sentAntecedentMessage(i) += currFactorValue * receivedPropertyMessage(j) * receivedAntecedentPropertyMessage(k);
//          sentAntecedentPropertyMessage(k) += currFactorValue * receivedPropertyMessage(j) * receivedAntecedentMessage(i);
//          k += 1;
//        }
//      }
//    }
    
    GUtil.normalizeiHard(sentPropertyMessage);
    GUtil.normalizeiHard(sentAntecedentMessage);
    GUtil.normalizeiHard(sentAntecedentPropertyMessage);
    
    if (sentPropertyMessage.contains(0.0)) {
      Logger.logss("Received prop message: " + receivedPropertyMessage.toSeq);
      Logger.logss("Received antecedent prop message: " + receivedAntecedentPropertyMessage.toSeq);
      Logger.logss("Received antecedent message: " + receivedAntecedentMessage.toSeq);
      for (i <- 0 until antecedentNode.domain.size) {
        for (j <- 0 until propertyNode.domain.size) {
          // While loop for the inner loop here
          var k = 0;
          while (k < antecedentPropertyNode.domain.size) {
            Logger.logss("Factor value for " + j + " " + i + " " + k + ": " + factorValue(j, i, k));
            k += 1;
          }
        }
      }
      require(false);
    }
    require(!sentAntecedentMessage.contains(0.0));
    require(!sentAntecedentPropertyMessage.contains(0.0));
    propertyNode.receiveMessage(this, sentPropertyMessage);
    antecedentNode.receiveMessage(this, sentAntecedentMessage);
    antecedentPropertyNode.receiveMessage(this, sentAntecedentPropertyMessage);
  }
  
  def getAllAssociatedFeatures(): Array[String] = new Array[String](0);
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {}
}


// Old unary factor implementation, only used for antecedent nodes anymore
class UnaryFactorOld(val propertyNode: Node[_]) extends Factor {
//                     val unaryFactor: Array[Double]) extends Factor {
  val unaryFactor = Array.fill(propertyNode.domain.size)(0.0);
  propertyNode.registerFactor(this);
  var receivedPropertyMessage: Array[Double] = null;
  
  def setWeights(newWeights: Array[Double]) {
    // Do nothing
  }
  
  def setUnaryFactor(newFactor: Array[Double]) {
    require(unaryFactor.length == newFactor.length);
    for (i <- 0 until unaryFactor.length) {
      unaryFactor(i) = newFactor(i);
    }
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    if (node == propertyNode) {
      receivedPropertyMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  def sendMessages() {
    propertyNode.receiveMessage(this, unaryFactor);
  }
  
  def getAllAssociatedFeatures(): Array[String] = new Array[String](0);
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {
    // No features so nothing to do
  }
}

class UnaryFactor(val propertyNode: Node[_],
                  val specifiedValue: String,
                  val agreeFeature: String,
                  val disagreeFeature: String,
                  val featurizer: PairwiseIndexingFeaturizer) extends Factor {
  var cachedWeights: Array[Double] = null;
  propertyNode.registerFactor(this);
  var receivedPropertyMessage: Array[Double] = null;
  var cachedMessage: Array[Double] = new Array[Double](propertyNode.domain.size);
  var cacheDirty = true;
  
  val agreeFeatureIndex = featurizer.getIndex(agreeFeature, false);
  val disagreeFeatureIndex = featurizer.getIndex(disagreeFeature, false);
  require(agreeFeatureIndex < featurizer.getIndexer.size);
  require(disagreeFeatureIndex < featurizer.getIndexer.size);
  
  def setWeights(newWeights: Array[Double]) {
    this.cachedWeights = newWeights;
    this.cacheDirty = true;
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    if (node == propertyNode) {
      receivedPropertyMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  
  def factorValue(propertyValueIdx: Int): Double = {
    if (propertyNode.domain.entries(propertyValueIdx) == specifiedValue) {
      Math.exp(cachedWeights(agreeFeatureIndex))
    } else {
      Math.exp(cachedWeights(disagreeFeatureIndex))
    }
  }
  
  def sendMessages() {
    if (cacheDirty) {
      for (i <- 0 until cachedMessage.size) {
        cachedMessage(i) = factorValue(i);
      }
      cacheDirty = false;
    }
    propertyNode.receiveMessage(this, cachedMessage);
  }
  
  def getAllAssociatedFeatures(): Array[String] = {
    Seq(agreeFeature, disagreeFeature).toArray;
  }
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {
    var agreeUnnormalizedProb = 0.0;
    var disagreeUnnormalizedProb = 0.0;
    var normalizer = 0.0;
    for (i <- 0 until propertyNode.domain.size) {
      val value = factorValue(i) * receivedPropertyMessage(i);
      // Accumulate expected agreement and disagreement counts
      if (propertyNode.domain.entries(i) == specifiedValue) {
        agreeUnnormalizedProb += value;
      } else {
        disagreeUnnormalizedProb += value;
      }
      normalizer += value;
    }
    gradient(agreeFeatureIndex) += scale * agreeUnnormalizedProb/normalizer;
    gradient(disagreeFeatureIndex) += scale * disagreeUnnormalizedProb/normalizer;
  }
}

class UnaryFactorGeneral(val propertyNode: Node[_],
                         val features: Array[Seq[String]],
                         val indexedFeatures: Array[Seq[Int]],
                         val defaultValues: Array[Double]) extends Factor {
  var cachedWeights: Array[Double] = null;
  propertyNode.registerFactor(this);
  require(propertyNode.domain.size == features.size);
  var receivedPropertyMessage: Array[Double] = null;
  var cachedMessage = new Array[Double](propertyNode.domain.size);
  var cacheDirty = true;
  
//  val indexedFeaturesForEachValue = featuresForEachValue.map(_.map(featurizer.getIndex(_, false)));
  
  def setWeights(newWeights: Array[Double]) {
    this.cachedWeights = newWeights;
    this.cacheDirty = true;
  }
  
  def receiveMessage(node: Node[_], message: Array[Double]) {
    if (node == propertyNode) {
      receivedPropertyMessage = message;
    } else {
      throw new RuntimeException("Bad node in graph");
    }
  }
  
  def factorValue(propertyValueIdx: Int): Double = {
    val exponent = defaultValues(propertyValueIdx) + indexedFeatures(propertyValueIdx).foldLeft(0.0)((currVal: Double, featIdx: Int) => currVal + cachedWeights(featIdx));
    Math.exp(exponent);
  }
  
  def sendMessages() {
    if (cacheDirty) {
      for (i <- 0 until cachedMessage.size) {
        cachedMessage(i) = factorValue(i);
      }
      cacheDirty = false;
    }
    propertyNode.receiveMessage(this, cachedMessage);
  }
  
  def getAllAssociatedFeatures(): Array[String] = {
    features.flatMap((feats: Seq[String]) => feats).toSet.toArray;
  }
  
  def addExpectedFeatureCounts(scale: Double, gradient: Array[Double]) {
    val marginalProbsUnnormalized = new Array[Double](propertyNode.domain.size);
    var normalizer = 0.0;
    for (i <- 0 until propertyNode.domain.size) {
      val value = factorValue(i) * receivedPropertyMessage(i);
      marginalProbsUnnormalized(i) = value;
      normalizer += value;
    }
    for (i <- 0 until propertyNode.domain.size) {
      for (indexedFeat <- indexedFeatures(i)) {
        gradient(indexedFeat) += scale * marginalProbsUnnormalized(i)/normalizer;
      }
    }
  }
}
