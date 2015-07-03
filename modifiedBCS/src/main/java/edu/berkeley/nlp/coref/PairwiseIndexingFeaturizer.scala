package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.JavaConverters._
import edu.berkeley.nlp.coref.sem.QueryCountsBundle

trait PairwiseIndexingFeaturizer {
  
  def getIndexer(): Indexer[String];

  def getIndex(feature: String, addToFeaturizer: Boolean): Int;
  
  def getQueryCountsBundle: QueryCountsBundle;

  def featurizeIndex(docGraph: DocumentGraph, currMentIdx: Int, antecedentIdx: Int, addToFeaturizer: Boolean): Seq[Int];
  
  def printFeatureTemplateCounts() {
    val indexer = getIndexer();
    val templateCounts = new Counter[String]();
    for (i <- 0 until indexer.size) {
      val currFeatureName = indexer.get(i);
      val currFeatureTemplateStop = currFeatureName.indexOf("=");
      if (currFeatureTemplateStop == -1) {
        Logger.logss("No =: " + currFeatureName);
      } else {
        templateCounts.incrementCount(currFeatureName.substring(0, currFeatureTemplateStop), 1.0);
      }
    }
    templateCounts.keepTopNKeys(200);
    if (templateCounts.size > 200) {
      Logger.logss("Not going to print more than 200 templates");
    }
    templateCounts.keySet().asScala.toSeq.sorted.foreach(template => Logger.logss(template + ": " + templateCounts.getCount(template).toInt));
  }
}