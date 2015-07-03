package edu.berkeley.nlp.coref

case class PruningStrategy(val strategy: String) {
  
  def getDistanceArgs(): (Int, Int) = {
    require(strategy.startsWith("distance"));
    val splitStrategy = strategy.split(":");
    (splitStrategy(1).toInt, splitStrategy(2).toInt);
  }
  
  def getLogRatio(): Double = {
    require(strategy.startsWith("c2flogratio"));
    strategy.substring(strategy.indexOf(":") + 1).toDouble;
  }
}