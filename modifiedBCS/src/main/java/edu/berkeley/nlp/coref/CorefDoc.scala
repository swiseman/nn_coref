package edu.berkeley.nlp.coref
import java.io.File

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.JavaConverters.mapAsScalaMapConverter
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import edu.berkeley.nlp.coref.lang.Language
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger

case class CorefDoc(val rawDoc: ConllDoc,
                    val goldMentions: Seq[Mention],
                    val goldClustering: OrderedClustering,
                    val predMentions: Seq[Mention]) {
  
  var oraclePredOrderedClustering: OrderedClustering = null;
  
  def numPredMents = predMentions.size;
  
  /**
   * Determines and caches an "oracle predicted clustering." For each predicted mention:
   * --If that mention does not have a corresponding gold mention (start and end indices match):
   *   --Put the current mention in its own cluster.
   * --If that mention does have a corresponding gold mention:
   *   --Fetch that mention's antecedents (if any)
   *   --Choose the first with a corresponding predicted mention (if any)
   *   --Assign this mention as the current mention's parent.
   */
  def getOraclePredClustering = {
    if (oraclePredOrderedClustering == null) {
      val predToGoldIdxMap = new HashMap[Int,Int]();
      val goldToPredIdxMap = new HashMap[Int,Int]();
      for (pIdx <- 0 until predMentions.size) {
        for (gIdx <- 0 until goldMentions.size) {
          val pMent = predMentions(pIdx);
          val gMent = goldMentions(gIdx);
          if (pMent.sentIdx == gMent.sentIdx && pMent.startIdx == gMent.startIdx && pMent.endIdx == gMent.endIdx) {
            predToGoldIdxMap.put(pIdx, gIdx);
            goldToPredIdxMap.put(gIdx, pIdx);
          }
        }
      }
      val oracleClusterIds = new ArrayBuffer[Int];
      var nextClusterId = 0;
      for (predIdx <- 0 until predMentions.size) {
        // Fetch the parent
        var parent = -1;
        if (predToGoldIdxMap.contains(predIdx)) {
          val correspondingGoldIdx = predToGoldIdxMap(predIdx);
          // Find the antecedents of the corresponding gold mention
          val goldAntecedentIdxs = goldClustering.getAllAntecedents(correspondingGoldIdx);
          // For each one, do a weird data sanitizing check, then try to find a corresponding
          // predicted mention to use as the predicted parent
          for (goldAntecedentIdx <- goldAntecedentIdxs.reverse) {
            val correspondingGold = goldMentions(correspondingGoldIdx);
            val goldAntecedent = goldMentions(goldAntecedentIdx);
            // wsj_0990 has some duplicate gold mentions, need to handle these...
            val sameMention = goldAntecedent.sentIdx == correspondingGold.sentIdx && goldAntecedent.startIdx == correspondingGold.startIdx && goldAntecedent.endIdx == correspondingGold.endIdx
            if (!sameMention && goldToPredIdxMap.contains(goldAntecedentIdx)) {
              val predAntecedentIdx = goldToPredIdxMap(goldAntecedentIdx)
              if (predAntecedentIdx >= predIdx) {
                val ment = predMentions(predIdx);
                val predAntecedent = predMentions(predAntecedentIdx);
                Logger.logss("Monotonicity violated:\n" +
                          "Antecedent(" + predAntecedentIdx + "): " + predAntecedent.startIdx + " " + predAntecedent.endIdx + " " + predAntecedent.headIdx + "\n" +
                          "Current(" + predMentions.indexOf(ment) + "): " + ment.startIdx + " " + ment.endIdx + " " + ment.headIdx + "\n" +
                          "Gold antecedent(" + goldMentions.indexOf(goldAntecedent) + "): " + goldAntecedent.startIdx + " " + goldAntecedent.endIdx + " " + goldAntecedent.headIdx + "\n" +
                          "Gold current(" + goldMentions.indexOf(correspondingGold) + "): " + correspondingGold.startIdx + " " + correspondingGold.endIdx + " " + correspondingGold.headIdx);
                Logger.logss("Setting parent to -1...");
                parent = -1;
              } else {
                parent = predAntecedentIdx
              }
            }
          }
        }
        // Now compute the oracle cluster ID
        val clusterId = if (parent == -1) {
          nextClusterId += 1;
          nextClusterId - 1;
        } else {
          oracleClusterIds(parent);
        }
        oracleClusterIds += clusterId;
      }
      oraclePredOrderedClustering = OrderedClustering.createFromClusterIds(oracleClusterIds);
    }
    oraclePredOrderedClustering
  }
}

object CorefDoc {
  
  def checkGoldMentionRecall(docs: Seq[CorefDoc]) {
    var numGMs = docs.map(_.goldMentions.size).reduce(_ + _);
    val numPMs = docs.map(_.predMentions.size).reduce(_ + _);
    val numNomPMs = docs.map(doc => doc.predMentions.filter(_.mentionType == MentionType.NOMINAL).size).reduce(_ + _);
    val numPropPMs = docs.map(doc => doc.predMentions.filter(_.mentionType == MentionType.PROPER).size).reduce(_ + _);
    val numPronPMs = docs.map(doc => doc.predMentions.filter(_.mentionType == MentionType.PRONOMINAL).size).reduce(_ + _);
    var numGMsRecalled = 0;
    var numGMsUnrecalledNonConstituents = 0;
    for (doc <- docs; gm <- doc.goldMentions) {
      if (doc.predMentions.filter(pm => pm.startIdx == gm.startIdx && pm.endIdx == gm.endIdx).size >= 1) {
        numGMsRecalled += 1;
      } else {
        if (!doc.rawDoc.trees(gm.sentIdx).isConstituent(gm.startIdx, gm.endIdx)) {
          numGMsUnrecalledNonConstituents += 1;
        }
      }
    }
    Logger.logss("Detected " + numPMs + " predicted mentons (" + numNomPMs + " nominal, " + numPropPMs + " proper, " + numPronPMs + " pronominal), " +
                 numGMsRecalled + " / " + numGMs + " = " + (numGMsRecalled.toDouble/numGMs) + " gold mentions recalled (" + numGMsUnrecalledNonConstituents + " missed ones are not constituents)")
  }
}