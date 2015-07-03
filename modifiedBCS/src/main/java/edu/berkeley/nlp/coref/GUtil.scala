package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Iterators
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.math.SloppyMath
import scala.util.Sorting
import java.util.Collection

object GUtil {
  
  def fmt(mat: Array[Array[Double]]): String = {
    var str = "";
    for (i <- 0 until mat.size) {
      for (j <- 0 until mat(i).size) {
        str += GUtil.fmt(mat(i)(j)) + "\t";
      }
      str += "\n";
    }
    str;
  }
  
//  def fmt(col: Collection[Double]): String = {
//    if (col.size == 0) {
//      "[]"
//    } else {
//      "[" + col.foldLeft("")((curr, nextD) => curr + fmt(nextD) + ", ").dropRight(2) + "]";
//    }
//  }
  
  def fmt(d: Double): String = {
    if (d.isNaN) {
      "NaN";
    } else if (d.isPosInfinity) {
      "+Inf";
    } else if (d.isNegInfinity) {
      "-Inf";
    } else {
      if (d < 0) "-" + fmtPositiveNumber(-d) else fmtPositiveNumber(d);
    }
  }
  
  def fmtProb(d: Double): String = {
    fmtPositiveNumber(d);
  }
  
  def fmtPositiveNumber(d: Double): String = {
    require(d >= 0);
    if (d == 0) {
      "0";
    }
    if (d < 1e-20) {
      "tiny"
    } else if (d < 0.001) {
      val numPlacesToMove = Math.ceil(-Math.log(d)/Math.log(10)).toInt;
      "%1.1f".format(d * Math.pow(10, numPlacesToMove)) + "e-" + numPlacesToMove; 
    } else if (d < 10000) {
      "%1.3f".format(d);
    } else {
      val numPlacesToMove = Math.floor(Math.log(d)/Math.log(10)).toInt;
      "%1.1f".format(d / Math.pow(10, numPlacesToMove)) + "e" + numPlacesToMove;
    }
  }
  
  def fmtTwoDigitNumber(d: Double, numDecimalPlaces: Int): String = {
    ("%1." + numDecimalPlaces + "f").format(d);
  }
  
  def containsNaN(array: Array[Double]): Boolean = {
    var containsNaN = false;
    for (value <- array) {
      containsNaN = containsNaN || value.isNaN;
    }
    containsNaN;
  }
  
  def containsNaNOrNegInf(array: Array[Double]): Boolean = {
    var bad = false;
    for (value <- array) {
      bad = bad || value.isNaN || value.isNegInfinity;
    }
    bad;
  }

  def getNBest[A](stuff: Seq[A], scorer: (A) => Double, n: Int): Seq[(A, Double)] = {
    val counter = new Counter[A]();
    for (thing <- stuff) {
      counter.setCount(thing, scorer(thing));
    }
    val results = new ArrayBuffer[(A, Double)]();
    for (thing <- Iterators.able(counter.asPriorityQueue()).asScala) {
      if (results.size < n) {
        results += new Tuple2(thing, counter.getCount(thing));
      }
    }
    results;
  }
  
  def getTopNKeysSubCounter(counter: Counter[String], n: Int) = {
    val newCounter = new Counter[String]();
    val pq = counter.asPriorityQueue()
    var numPrinted = 0;
    while (pq.hasNext() && numPrinted < n) {
      val obj = pq.next();
      newCounter.setCount(obj, counter.getCount(obj));
      numPrinted += 1;
    }
    newCounter;
  }
  
  def normalizeiSoft(arr: Array[Double]): Boolean = {
    var idx = 0;
    var total = 0.0;
    while (idx < arr.size) {
      total += arr(idx);
      idx += 1;
    }
    if (total <= 0.0) {
      false;
    } else {
      idx = 0;
      while (idx < arr.size) {
        arr(idx) /= total;
        idx += 1;
      }
      true;
    }
  }
  
  def normalizeiHard(arr: Array[Double]) {
    var idx = 0;
    var total = 0.0;
    while (idx < arr.size) {
      total += arr(idx);
      idx += 1;
    }
    if (total <= 0.0) {
      throw new RuntimeException("Bad total for normalizing: " + total);
    }
    idx = 0;
    while (idx < arr.size) {
      arr(idx) /= total;
      idx += 1;
    }
  }
  
  def expAndNormalizeiHard(arr: Array[Double]) {
    var idx = 0;
    while (idx < arr.size) {
      arr(idx) = Math.exp(arr(idx));
      idx += 1;
    }
    normalizeiHard(arr);
  }
  
  def renderMat[A](mat: Array[Array[A]]): String = {
    mat.map(row => row.map(_.toString).reduce((c1, c2) => c1 + ", " + c2)).reduce((r1, r2) => r1 + "\n" + r2);
  }
  
  def normalizei(vector: Array[Double]) {
    val normalizer = vector.reduce(_ + _);
    for (i <- 0 until vector.size) {
      vector(i) /= normalizer;
    }
  }
  
  def logNormalizei(vector: Array[Double]) {
    val normalizer = SloppyMath.logAdd(vector);
    for (i <- 0 until vector.size) {
      vector(i) -= normalizer;
    }
  }
  
  def logNormalizeiByRow(mat: Array[Array[Double]]) {
    for (i <- 0 until mat.size) {
      val normalizer = SloppyMath.logAdd(mat(i));
      for (j <- 0 until mat(i).size) {
        mat(i)(j) -= normalizer;
      }
    }
  }
  
  def computeQuantile(nums: Array[Double], quantile: Double): Double = {
    val numsCpy = new Array[Double](nums.size);
    Array.copy(nums, 0, numsCpy, 0, nums.size);
    Sorting.quickSort(numsCpy);
    numsCpy((quantile * nums.size).toInt);
  }
  
  def main(args: Array[String]) {
    println(fmtProb(1.0));
    println(fmtProb(0.01));
    println(fmtProb(0.001));
    println(fmtProb(0.0001));
    println(fmtProb(0.00001));
    println(fmtProb(0.000001));
    println(fmtProb(0.0000001));
    
    println(fmtProb(0.000000000000000000000001));
  }
}