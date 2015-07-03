package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.fig.prob.Dirichlet

object OraclePosteriorSampler {
  
  def sample(alphas: Array[Double], rng: java.util.Random): Array[Double] = {
    new Dirichlet(alphas).sample(rng);
  }

  def randomPosterior(domainSize: Int, specialIndex: Int, rng: java.util.Random): Array[Double] = {    
    val baseAlpha = 1.0;
    val specialAlpha = if (domainSize == 2) {
      2.1
    } else if (domainSize == 5) {
      3.5
    } else {
      throw new RuntimeException("Domain size " + domainSize + " doesn't have fitparameters");
    }
    val alphas = Array.fill(domainSize)(baseAlpha);
    alphas(specialIndex) = specialAlpha;
    sample(alphas, rng);
  }
  
  def main(args: Array[String]) {
    val rng = new java.util.Random(0);
//    val alpha = 0.1;
//    val specialAlpha = 0.3;
    {
      //    val alpha = 0.4;
      //    val specialAlpha = 1.0;
      val alpha = 1.0;
      val specialAlpha = 2.1;
      val totalSamples = 1000;
      var numInversions = 0;
      var totalInverted = 0.0;
      var totalNoninverted = 0.0;
      for (i <- 0 until totalSamples) {
        val currSample = sample(Array(specialAlpha, alpha), rng).toSeq;
        val max = currSample.reduce(Math.max(_, _));
        if (currSample(0) < max - 1e-8) {
          numInversions += 1;
          totalInverted += max;
        } else {
          totalNoninverted += max;
        }
      }
      println("Domain size 2");
      println("Num inversions: " + numInversions + "/" + totalSamples);
      println("Avg max if not inverted: " + totalNoninverted/(totalSamples - numInversions));
      println("Avg max if inverted: " + totalInverted/numInversions);
    }
    
    {
      //    val alpha = 0.4;
      //    val specialAlpha = 1.9;
      val alpha = 1.0;
      val specialAlpha = 3.5;
      val totalSamples = 1000;
      var numInversions = 0;
      var totalInverted = 0.0;
      var totalNoninverted = 0.0;
      for (i <- 0 until totalSamples) {
        val currSample = sample(Array(specialAlpha, alpha, alpha, alpha, alpha), rng).toSeq;
        val max = currSample.reduce(Math.max(_, _));
        if (currSample(0) < max - 1e-8) {
          numInversions += 1;
          totalInverted += max;
        } else {
          totalNoninverted += max;
        }
      }
      println("Domain size 5");
      println("Num inversions: " + numInversions + "/" + totalSamples);
      println("Avg max if not inverted: " + totalNoninverted/(totalSamples - numInversions));
      println("Avg max if inverted: " + totalInverted/numInversions);
    }
  }
}