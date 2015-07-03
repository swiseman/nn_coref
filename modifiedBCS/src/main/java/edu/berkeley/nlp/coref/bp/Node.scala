package edu.berkeley.nlp.coref.bp
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.GUtil

class Node[T](val domain: Domain[T]) {
  var factors = new ArrayBuffer[Factor]();
  var receivedMessages: Array[Array[Double]] = null;
  var sentMessages: Array[Array[Double]] = null;
  var cachedBeliefsOrMarginals: Array[Double] = Array.fill(domain.size)(0.0);
  
  def registerFactor(factor: Factor) {
    factors += factor;
  }
  
  // TODO: Do I need this null thing?
  def initializeReceivedMessagesUniform() {
    if (receivedMessages == null) {
      receivedMessages = new Array[Array[Double]](factors.size);
    } else {
      for (i <- 0 until receivedMessages.size) {
        receivedMessages(i) = null;
      }
    }
  }
  
  // This is just here so we can let things be null...At some point, it was a problem because
  // the received messages remember which factors sent them, so clearing them for some reason
  // caused problems (maybe writing the value 1.0 was problematic when we weren't clearing the
  // received messages on the other end?). Can probably get rid of this somehow and just do the
  // obvious thing of initializing messages to 1.0.
  def receivedMessageValue(i: Int, j: Int): Double = {
    if (receivedMessages(i) == null) {
      1.0;
    } else {
      receivedMessages(i)(j);
    }
  }
  
  def receiveMessage(factor: Factor, message: Array[Double]) {
    require(receivedMessages != null);
    require(!GUtil.containsNaN(message));
    val idx = factors.indexOf(factor);
    require(idx != -1 && idx < receivedMessages.size);
    if (message.toSeq.contains(0.0)) {
      Logger.logss("For domain: " + domain + ", bad received message: " + message.toSeq + " from " + factor.getClass());
      Logger.logss("Previous message: " + receivedMessages(factors.indexOf(factor)).toSeq);
      require(false);
    }
    if (message.reduce(_ + _) == 0) {
      Logger.logss("For domain: " + domain + ", bad received message: " + message.toSeq + " from " + factor.getClass());
      Logger.logss("Previous message: " + receivedMessages(factors.indexOf(factor)).toSeq);
      require(false);
    }
    require(message.size == domain.size);
    receivedMessages(factors.indexOf(factor)) = message;
  }
  
  def sendMessages() {
//    sendMessagesUseRealSpace();
    sendMessagesUseLogSpace();
  }
  
  def sendMessagesUseRealSpace() {
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      cachedBeliefsOrMarginals(i) = 1.0;
    }
    require(receivedMessages.size == factors.size);
    for (i <- 0 until receivedMessages.size) {
      var j = 0;
      while (j < cachedBeliefsOrMarginals.size) {
        cachedBeliefsOrMarginals(j) *= receivedMessageValue(i, j);
        j += 1;
      }
    }
    // Normalize beliefs
    val normalizedNonzero = GUtil.normalizeiSoft(cachedBeliefsOrMarginals);
    if (!normalizedNonzero) {
      Logger.logss("For domain: " + domain + ", received messages:" + receivedMessages.foldLeft("")((currStr, msg) => currStr + "\n" + msg.toSeq.toString))
      require(false);
    }
    if (sentMessages == null) {
      sentMessages = new Array[Array[Double]](factors.size);
    }
    for (i <- 0 until factors.length) {
      sentMessages(i) = new Array[Double](domain.size);
      var j = 0;
      while (j < domain.size) {
        val rmVal = receivedMessageValue(i, j);
        if (rmVal == 0) {
          sentMessages(i)(j) = 0;
        } else {
          sentMessages(i)(j) = cachedBeliefsOrMarginals(j)/rmVal;
        }
        j += 1;
      }
      factors(i).receiveMessage(this, sentMessages(i));
    }
  }
  
  def sendMessagesUseLogSpace() {
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      cachedBeliefsOrMarginals(i) = 0.0;
    }
    require(receivedMessages.size == factors.size);
    for (i <- 0 until receivedMessages.size) {
      var j = 0;
      while (j < cachedBeliefsOrMarginals.size) {
        cachedBeliefsOrMarginals(j) += Math.log(receivedMessageValue(i, j));
        j += 1;
      }
    }
    GUtil.logNormalizei(cachedBeliefsOrMarginals);
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      cachedBeliefsOrMarginals(i) = Math.exp(cachedBeliefsOrMarginals(i));
    }
    if (sentMessages == null) {
      sentMessages = new Array[Array[Double]](factors.size);
    }
    for (i <- 0 until factors.length) {
      sentMessages(i) = new Array[Double](domain.size);
      var j = 0;
      while (j < domain.size) {
        val rmVal = receivedMessageValue(i, j);
        if (rmVal == 0) {
          sentMessages(i)(j) = 0;
        } else {
          sentMessages(i)(j) = cachedBeliefsOrMarginals(j)/rmVal;
        }
        j += 1;
      }
      factors(i).receiveMessage(this, sentMessages(i));
    }
  }
  
  def getMarginals(): Array[Double] = {
    getMarginalsUseLogSpace();
  }
  
  def getMarginalsUseLogSpace(): Array[Double] = {
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      cachedBeliefsOrMarginals(i) = 0.0;
    }
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      for (j <- 0 until receivedMessages.size) {
        cachedBeliefsOrMarginals(i) += Math.log(receivedMessageValue(j, i));
      }
    }
    GUtil.logNormalizei(cachedBeliefsOrMarginals);
    for (i <- 0 until cachedBeliefsOrMarginals.size) {
      cachedBeliefsOrMarginals(i) = Math.exp(cachedBeliefsOrMarginals(i));
    }
    cachedBeliefsOrMarginals
  }
}