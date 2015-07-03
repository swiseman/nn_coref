package edu.berkeley.nlp.coref.sem

import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.coref.MentionType
import edu.berkeley.nlp.util.Counter
import edu.berkeley.nlp.coref.PronounDictionary
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.Mention

object QueryCountAnalyzer {
  
  def renderSomeQueries(docGraph: DocumentGraph, predBackptrs: Seq[Int], queryCounts: QueryCountsBundle) = {
    var rendered = "";
    for (mentIdx <- 0 until docGraph.size) {
      if (isReferring(docGraph, mentIdx) &&
          isIncorrect(docGraph, mentIdx, predBackptrs(mentIdx)) &&
          hasReferringAntecedents(docGraph, mentIdx) &&
          !hasHeadMatchWithAntecedent(docGraph, mentIdx)) {
        val myHeadTc = docGraph.getMention(mentIdx).headString;
        val antIndicesCounts = (0 until mentIdx).filter(idx => isReferring(docGraph, idx)).map(idx => (idx, queryCounts.pairCounts.getCount(myHeadTc, docGraph.getMention(idx).headString)))
        // Top five scores and whether they're in
        val topAntIndicesCounts = antIndicesCounts.sortBy(_._2).reverse.slice(0, Math.min(5, antIndicesCounts.size));
        val goldRefAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(mentIdx).filter(docGraph.getMention(_).mentionType != MentionType.PRONOMINAL);
        def renderMentionAndCount = (idx: Int, count: Double) => "[" + idx + ": " + renderMentionWithHead(docGraph.getMention(idx)) + ", " + count + ", " + 
                            (if (docGraph.getGoldAntecedentsUnderCurrentPruning(mentIdx).contains(idx)) "corr" else "wrong") + "] ";
        rendered += docGraph.corefDoc.rawDoc.docID + " " + mentIdx + ": " + renderMentionWithHead(docGraph.getMention(mentIdx)) + "\n  correct = ";
        for (goldRefAntecedent <- goldRefAntecedents) {
          val antHeadTc = docGraph.getMention(goldRefAntecedent).headString;
          rendered += renderMentionAndCount(goldRefAntecedent, queryCounts.pairCounts.getCount(myHeadTc, antHeadTc));
        }
        rendered += "\n  top five = ";
        for (i <- 0 until topAntIndicesCounts.size) {
          rendered += renderMentionAndCount(topAntIndicesCounts(i)._1, topAntIndicesCounts(i)._2);
        }
        rendered += "\n";
      }
    }
    rendered;
  }
  
  private def renderMentionWithHead(mention: Mention) = {
    val startIdx = Math.max(mention.headIdx - mention.startIdx - 2, 0);
    val endIdx = Math.min(mention.headIdx - mention.startIdx + 3, mention.words.size);
    val str = mention.words.slice(startIdx, mention.headIdx - mention.startIdx).foldLeft("")(_ + " " + _) + " _" + mention.words(mention.headIdx - mention.startIdx) +
      "_" + mention.words.slice(mention.headIdx - mention.startIdx + 1, endIdx).foldLeft("")(_ + " " + _)
    str.trim;
  }

  def renderQueryCountStats(docGraphs: Seq[DocumentGraph], allPredBackptrs: Seq[Seq[Int]], queryCounts: QueryCountsBundle) = {
//    var numTop = 0.0;
//    var numUnseen = 0.0;
//    for (i <- 0 until docGraphs.size) {
//      val docGraph = docGraphs(i);
//      for (j <- 0 until docGraph.size) {
//        if (isReferring(docGraph, j) &&
//            isIncorrect(docGraph, j, allPredBackptrs(i)(j)) &&
//            hasReferringAntecedents(docGraph, j) &&
//            !hasHeadMatchWithAntecedent(docGraph, j)) {
//          val myHeadTc = docGraph.getMention(j).headString;
//          val antHeads = (0 until j).filter(isReferring(docGraph, _)).map(docGraph.getMention(_).headString);
//          val topCountScore = queryCounts.pairCounts.getCount(myHeadTc, )
//        }
//      }
//    }
    ""
  }
  
  def renderTopFailedRecallHeadPairs(docGraphs: Seq[DocumentGraph], allPredBackptrs: Array[Array[Int]]) = {
    val headCounter = new Counter[String]();
    val headCounterMislead = new Counter[String]();
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      for (j <- 0 until docGraph.size) {
        if (isReferring(docGraph, j) &&
            isIncorrect(docGraph, j, allPredBackptrs(i)(j)) &&
            hasReferringAntecedents(docGraph, j) &&
            hasHeadMatchWithAntecedent(docGraph, j)) {
          if (hasHeadMatchWithPrediction(docGraph, j, allPredBackptrs(i)(j))) {
            headCounterMislead.incrementCount(docGraph.getMention(j).headStringLc, 1.0);
          }
          headCounter.incrementCount(docGraph.getMention(j).headStringLc, 1.0);
        }
      }
    }
    var rendered = headCounter.size + " heads missed, " + headCounterMislead.size + " heads mislead\n";
    headCounter.keepTopNKeys(100);
    rendered += headCounter.toString + "\n";
    headCounterMislead.keepTopNKeys(100);
    rendered += headCounterMislead.toString + "\n";
    rendered;
  }
  
  // N.B. Referring here means nominal or proper, not coreferent
  def isReferring(docGraph: DocumentGraph, idx: Int) = {
    docGraph.getMention(idx).mentionType != MentionType.PRONOMINAL;
  }
  
  def isIncorrect(docGraph: DocumentGraph, idx: Int, backptr: Int) = {
    !docGraph.getGoldAntecedentsUnderCurrentPruning(idx).contains(backptr);
  }
  
  def isPredictedNewCluster(docGraph: DocumentGraph, idx: Int, backptr: Int) = {
    backptr == idx;
  }
  
  def hasReferringAntecedents(docGraph: DocumentGraph, idx: Int) = {
    val goldAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(idx);
    goldAntecedents.filter(i => docGraph.getMention(i).mentionType != MentionType.PRONOMINAL).size > 0;
  }
  
  def hasHeadMatchWithAntecedent(docGraph: DocumentGraph, idx: Int) = {
    val goldAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(idx);
    goldAntecedents.filter(i => docGraph.getMention(i).headStringLc == docGraph.getMention(idx).headStringLc).size > 0;
  }
  
  def hasHeadMatchWithPrediction(docGraph: DocumentGraph, idx: Int, backptr: Int) = {
    backptr != idx && docGraph.getMention(idx).headStringLc == docGraph.getMention(backptr).headStringLc;
  }
}