package edu.berkeley.nlp.coref.nchains
import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.GUtil
import scala.collection.mutable.HashMap
import edu.berkeley.nlp.coref.OrderedClustering
import scala.collection.JavaConverters._
import edu.berkeley.nlp.coref.Mention
import edu.berkeley.nlp.coref.MentionType
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Counter

object DiscourseAnalyzer {
  
  def renderGoldPredDiscourseAnalyses(doc: DocumentGraph, backptrs: Array[Int]) = {
    var str = "Gold analysis:\n" + renderGoldDiscourseAnalysis(doc);
    str += "\nPred analysis:\n" + renderPredDiscourseAnalysis(doc, backptrs);
    str;
  }
  
  def renderGoldDiscourseAnalysis(doc: DocumentGraph) = {
    renderDiscourseAnalysis(doc, doc.getGoldClustersNoPruning());
  }
  
  def renderPredDiscourseAnalysis(doc: DocumentGraph, backptrs: Array[Int]) = {
    renderDiscourseAnalysis(doc, convertBackptrs(doc, backptrs));
  }
  
  def convertBackptrs(doc: DocumentGraph, backptrs: Array[Int]): Seq[Seq[Mention]] = {
    val clusters = OrderedClustering.createFromBackpointers(backptrs).clusters;
    clusters.map(_.map(doc.getMention(_)));
  }
  
  // clusters are expected to be each internally ordered, 
  def renderDiscourseAnalysis(doc: DocumentGraph, clusters: Seq[Seq[Mention]]) = {
    var currSentNum = 0;
    var maxSentNum = doc.getMention(doc.size - 1).sentIdx;
    // Clusters should already be sorted by order of first appearance, due
    // to how they are indexed
//    for (goldCluster <- goldClusters) {
//      Logger.logss(goldCluster.first.getMentionTextLc() + ": " + goldCluster.map(_.sentNum));
//    }
    // Print the grid of sentences that each entity appears in, look for patterns
    val nonsingletonClusters = clusters.filter(_.size >= 2);
    val maxMentsSameClusterSameSentence = getMaxMentsSameClusterSameSentence(clusters);
    if (nonsingletonClusters.size == 0) {
      Logger.logss("WARNING: " + nonsingletonClusters.size + " nonsingleton clusters, " + clusters.size + " original clusters");
    }
    val occurrencesMat = for (i <- 0 until nonsingletonClusters.size) yield {
      val sentOccurrences: Array[String] = Array.tabulate(maxSentNum+1)(j => {
        var sentStr = "";
        for (ment <- nonsingletonClusters(i)) {
          if (ment.sentIdx == j) {
            if (ment.mentionType == MentionType.PROPER) {
              sentStr += "E";
            } else if (ment.mentionType == MentionType.NOMINAL) {
              sentStr += "N";
            } else {
              sentStr += "P";
            }
//            if (ment.isSubject) {
//              sentStr += "S";
//            } else if (ment.isDirectObject) {
//              sentStr += "D";
//            } else if (ment.isIndirectObject) {
//              sentStr += "O";
//            } else {
//              sentStr += "X";
//            }
          }
        }
        for (j <- sentStr.size until maxMentsSameClusterSameSentence) {
          sentStr += " ";
        }
        sentStr + "|";
      });
      sentOccurrences;
    };
    val occurrences = for (i <- 0 until nonsingletonClusters.size) yield {
      nonsingletonClusters(i).head.headString + ": " + occurrencesMat(i).foldLeft("")(_ + "" + _);
    }
    val maxLineLen = occurrences.map(_.size).foldLeft(0)(Math.max(_, _));
    var rendered = "Entity grid for " + doc.corefDoc.rawDoc.printableDocName + ":\n";
    for (occurrence <- occurrences) {
      val padSize = maxLineLen - occurrence.size;
      var pad = "";
      for (i <- 0 until padSize) {
        pad = pad + " ";
      }
      rendered += pad + occurrence + "\n";
    }
    rendered += "Entities per sentence counts:\n";
    val backreferenceCountBuckets = new Array[Int](11);
    for (j <- 0 until maxSentNum) {
      var currCount = 0;
      for (i <- 0 until nonsingletonClusters.size) {
        if (occurrencesMat(i)(j).trim != "") {
          currCount += 1;
        }
      }
      currCount = Math.min(currCount, 10);
      backreferenceCountBuckets(currCount) += 1;
    }
    rendered += backreferenceCountBuckets.toSeq;
    rendered;
  }
  
  def renderDiscourseErrorAnalysis(doc: DocumentGraph, predBackptrs: Array[Int]) = {
    val goldClusters = doc.getGoldClustersNoPruning();
    var maxSentNum = doc.getMention(doc.size - 1).sentIdx;
    val nonsingletonGoldClusters = goldClusters.filter(_.size >= 2);
    val maxMentsSameClusterSameSentence = getMaxMentsSameClusterSameSentence(goldClusters);
    val occurrencesMat = for (i <- 0 until nonsingletonGoldClusters.size) yield {
      val sentOccurrences: Array[String] = Array.tabulate(maxSentNum+1)(j => {
        var sentStr = "";
        for (ment <- nonsingletonGoldClusters(i)) {
          if (ment.sentIdx == j) {
            val mentIdx = ment.mentIdx;
            val errorType = getErrorType(doc, mentIdx, predBackptrs(mentIdx));
            sentStr += errorType;
          }
        }
        for (j <- sentStr.size until maxMentsSameClusterSameSentence) {
          sentStr += " ";
        }
        sentStr + "|";
      });
      sentOccurrences;
    };
    val occurrences = for (i <- 0 until nonsingletonGoldClusters.size) yield {
      nonsingletonGoldClusters(i).head.headString + ": " + occurrencesMat(i).foldLeft("")(_ + "" + _);
    }
    val maxLineLen = occurrences.map(_.size).foldLeft(0)(Math.max(_, _));
    var rendered = "Error analysis for " + doc.corefDoc.rawDoc.printableDocName + ":\n";
    for (occurrence <- occurrences) {
      val padSize = maxLineLen - occurrence.size;
      var pad = "";
      for (i <- 0 until padSize) {
        pad = pad + " ";
      }
      rendered += pad + occurrence + "\n";
    }
    rendered;
  }
  
  def renderAggregateDiscourseErrors(docs: Seq[DocumentGraph], allPredBackptrs: Array[Array[Int]]) = {
    val countsEachErrorType = new Array[Int](4);
    val countsEachErrorTypeEachDistance = Array.tabulate(4, 11)((i, j) => 0.0);
    val referringCountsEachErrorType = new Array[Int](4);
    val referringCountsEachErrorTypeEachDistance = Array.tabulate(4, 11)((i, j) => 0.0);
    val referringReferringCountsEachErrorTypeEachDistance = Array.tabulate(4, 11)((i, j) => 0.0);
//    val propertiesCounter = new Counter[String]
    val pairsPrevRef = new Counter[String]();
    val pairsFirstRef = new Counter[String]();
    val pairsPrevRefNoHeadMatch = new Counter[String]();
    val pairsFirstRefNoHeadMatch = new Counter[String]();
    val pairsBothRefNoHeadMatch = new Counter[String]();
    for (docIdx <- 0 until docs.size) {
      val doc = docs(docIdx);
      val predBackptrs = allPredBackptrs(docIdx);
      val goldClusters = doc.getGoldClustersNoPruning();
      val nonsingletonGoldClusters = goldClusters.filter(_.size >= 2);
      for (i <- 0 until nonsingletonGoldClusters.size) {
        for (mentClusterIdx <- 0 until nonsingletonGoldClusters(i).size) {
          val ment = nonsingletonGoldClusters(i)(mentClusterIdx);
          val mentIdx = ment.mentIdx;
          val referring = ment.mentionType != MentionType.PRONOMINAL;
          val errorType = getErrorType(doc, mentIdx, predBackptrs(mentIdx));
          val sentIdx = ment.sentIdx;
          if (mentClusterIdx > 0) {
            val prevSentIdx = nonsingletonGoldClusters(i)(mentClusterIdx - 1).sentIdx;
            val distance = Math.min(sentIdx - prevSentIdx, 10);
            countsEachErrorTypeEachDistance(errorType)(distance) += 1;
            if (referring) {
              referringCountsEachErrorTypeEachDistance(errorType)(distance) += 1;
            }
            val referringAntecedents = nonsingletonGoldClusters(i).slice(0, mentClusterIdx).filter(ment => ment.mentionType != MentionType.PRONOMINAL);
            if (errorType > 0 && !referringAntecedents.isEmpty) {
              val referringParent = referringAntecedents.last;
              val referringFirst = referringAntecedents.head;
              val distanceReferring = Math.min(sentIdx - referringParent.sentIdx, 10);
              if (referring) {
                referringReferringCountsEachErrorTypeEachDistance(errorType)(distanceReferring) += 1;
                pairsPrevRef.incrementCount(referringParent.headStringLc + "-" + ment.headStringLc, 1.0);
                pairsFirstRef.incrementCount(referringFirst.headStringLc + "-" + ment.headStringLc, 1.0);
                val headMatch = referringAntecedents.map(ant => ant.headStringLc == ment.headStringLc).reduce(_ || _);
                if (!headMatch) {
                  pairsPrevRefNoHeadMatch.incrementCount(referringParent.headStringLc + "-" + ment.headStringLc, 1.0);
                  pairsFirstRefNoHeadMatch.incrementCount(referringFirst.headStringLc + "-" + ment.headStringLc, 1.0);
                  if (referringAntecedents.size >= 2) {
                    pairsBothRefNoHeadMatch.incrementCount(referringFirst.headStringLc + "-" + referringParent.headStringLc + "-" + ment.headStringLc, 1.0);
                  }
                }
              }
            }
          }
          countsEachErrorType(errorType) += 1;
          if (referring) {
            referringCountsEachErrorType(errorType) += 1;
          }
        }
      }
    }
    var rendered = "";
    rendered += "ALL MENTIONS\n";
    rendered += countsEachErrorType.toSeq.toString + "\n";
    for (errType <- 0 until countsEachErrorTypeEachDistance.size) {
      rendered += countsEachErrorTypeEachDistance(errType).toSeq + "\n";
    }
    rendered += "REFERRING\n";
    rendered += referringCountsEachErrorType.toSeq.toString + "\n";
    for (errType <- 0 until referringCountsEachErrorTypeEachDistance.size) {
      rendered += referringCountsEachErrorTypeEachDistance(errType).toSeq + "\n";
    }
    rendered += "REFERRING-REFERRING\n";
    for (errType <- 0 until referringCountsEachErrorTypeEachDistance.size) {
      rendered += referringReferringCountsEachErrorTypeEachDistance(errType).toSeq + "\n";
    }
    rendered += "PAIR COUNTS\n";
    rendered += pairsPrevRef.size + " prev pairs (" + pairsPrevRef.totalCount + " tokens)\n";
    rendered += pairsFirstRef.size + " first pairs (" + pairsFirstRef.totalCount + " tokens)\n";
    rendered += pairsPrevRefNoHeadMatch.size + " prev pairs no head match (" + pairsPrevRefNoHeadMatch.totalCount + " tokens): " + GUtil.getTopNKeysSubCounter(pairsPrevRefNoHeadMatch, 500).toString + "\n";
    rendered += pairsFirstRefNoHeadMatch.size + " first pairs no head match (" + pairsFirstRefNoHeadMatch.totalCount + " tokens): " + GUtil.getTopNKeysSubCounter(pairsFirstRefNoHeadMatch, 500).toString + "\n";
    rendered += pairsBothRefNoHeadMatch.size + " both pairs no head match (" + pairsBothRefNoHeadMatch.totalCount + " tokens): " + GUtil.getTopNKeysSubCounter(pairsBothRefNoHeadMatch, 500).toString + "\n";
    for (errType <- 0 until referringCountsEachErrorTypeEachDistance.size) {
      rendered += referringReferringCountsEachErrorTypeEachDistance(errType).toSeq + "\n";
    }
    
    rendered;
  }
  
  def getMaxMentsSameClusterSameSentence(clusters: Seq[Seq[Mention]]) = {
    require(!clusters.isEmpty);
    clusters.map(cluster => {
      val map = new HashMap[Int,Int]();
      for (ment <- cluster) {
        if (!map.contains(ment.sentIdx)) {
          map.put(ment.sentIdx, 0);
        }
        map.put(ment.sentIdx, map(ment.sentIdx) + 1);
      }
      map.values.reduce(Math.max(_, _));
    }).reduce(Math.max(_, _));
  }
  
  def getErrorType(doc: DocumentGraph, mentIdx: Int, predAntIdx: Int) = {
    val oracleCluster = doc.getOraclePredClustering;
    if (oracleCluster.startsCluster(mentIdx) && mentIdx != predAntIdx) {
      2;
    } else if (!oracleCluster.startsCluster(mentIdx) && mentIdx == predAntIdx) {
      1;
    } else if (!oracleCluster.startsCluster(mentIdx) && !oracleCluster.areInSameCluster(mentIdx, predAntIdx)) {
      3;
    } else {
      0;
    };
  }
}