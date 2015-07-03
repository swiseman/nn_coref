package edu.berkeley.nlp.coref.preprocess
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer

case class NerExample(val words: Seq[String], val poss: Seq[String], val goldLabels: Seq[String], val labelIndexer: Indexer[String]) {

  def featurize(featureIndexer: Indexer[String], addToIndexer: Boolean): Array[Array[Array[Int]]] = {
    Array.tabulate(words.size, labelIndexer.size)((tokIdx, labelIdx) => {
      require(labelIndexer.size > labelIdx);
      val labelName = labelIndexer.getObject(labelIdx);
      // Extract word and word shape features 
      val wordAt = (i: Int) => if (tokIdx + i < 0) "<<START>>" else if (tokIdx + i >= words.size) "<<END>>" else words(tokIdx + i); 
      val wordShapeAt = (i: Int) => NerExample.shapeFor(wordAt(i));
      val wordClassAt = (i: Int) => NerExample.classFor(wordAt(i));
      val posAt = (i: Int) => if (tokIdx + i < 0) "<<START>>" else if (tokIdx + i >= words.size) "<<END>>" else poss(tokIdx + i);

      val feats = new ArrayBuffer[Int]();
      val maybeAddFeat = (feat: String) => {
        val labeledFeat = labelName + ":" + feat;
        if (addToIndexer || featureIndexer.contains(labeledFeat)) feats += featureIndexer.getIndex(labeledFeat)
      }
      // Words
      maybeAddFeat("-1W=" + wordAt(-2));
      maybeAddFeat("-1W=" + wordAt(-1));
      maybeAddFeat("0W=" + wordAt(0));
      maybeAddFeat("1W=" + wordAt(1));
      maybeAddFeat("2W=" + wordAt(2));
      // POS
      maybeAddFeat("-2P=" + posAt(-2));
      maybeAddFeat("-1P=" + posAt(-1));
      maybeAddFeat("0P=" + posAt(0));
      maybeAddFeat("1P=" + posAt(1));
      maybeAddFeat("2P=" + posAt(2));
      // Shape
      maybeAddFeat("-2S=" + wordShapeAt(-2));
      maybeAddFeat("-1S=" + wordShapeAt(-1));
      maybeAddFeat("0S=" + wordShapeAt(0));
      maybeAddFeat("1S=" + wordShapeAt(1));
      maybeAddFeat("2S=" + wordShapeAt(2));
      // Class
      maybeAddFeat("-2C=" + wordClassAt(-1));
      maybeAddFeat("-1C=" + wordClassAt(-1));
      maybeAddFeat("0C=" + wordClassAt(0));
      maybeAddFeat("1C=" + wordClassAt(1));
      maybeAddFeat("2C=" + wordClassAt(1));
      // POS-POS conjunctions
      maybeAddFeat("-2-1P=" + posAt(-2) + "," + posAt(-1));
      maybeAddFeat("-10P=" + posAt(-1) + "," + posAt(0));
      maybeAddFeat("01P=" + posAt(0) + "," + posAt(1));
      maybeAddFeat("12P=" + posAt(1) + "," + posAt(2));
      //        // Word-word conjunctions
      //        maybeAddFeat("-2-1W=" + wordAt(-2) + "," + wordAt(-1));
      //        maybeAddFeat("-10W=" + wordAt(-1) + "," + wordAt(0));
      //        maybeAddFeat("01W=" + wordAt(0) + "," + wordAt(1));
      //        maybeAddFeat("12W=" + wordAt(1) + "," + wordAt(2));
      // Word-POS conjunctions
      maybeAddFeat("-2-1PW=" + posAt(-2) + "," + wordAt(-1));
      maybeAddFeat("-10PW=" + posAt(-1) + "," + wordAt(0));
      maybeAddFeat("01PW=" + posAt(0) + "," + wordAt(1));
      maybeAddFeat("12PW=" + posAt(1) + "," + wordAt(2));
      maybeAddFeat("-2-1WP=" + wordAt(-2) + "," + posAt(-1));
      maybeAddFeat("-10WP=" + wordAt(-1) + "," + posAt(0));
      maybeAddFeat("01WP=" + wordAt(0) + "," + posAt(1));
      maybeAddFeat("12WP=" + wordAt(1) + "," + posAt(2));
      // Word-class conjunctions
      maybeAddFeat("-2-1CW=" + wordClassAt(-2) + "," + wordAt(-1));
      maybeAddFeat("-10CW=" + wordClassAt(-1) + "," + wordAt(0));
      maybeAddFeat("01CW=" + wordClassAt(0) + "," + wordAt(1));
      maybeAddFeat("12CW=" + wordClassAt(1) + "," + wordAt(2));
      maybeAddFeat("-2-1WC=" + wordAt(-2) + "," + wordClassAt(-1));
      maybeAddFeat("-10WC=" + wordAt(-1) + "," + wordClassAt(0));
      maybeAddFeat("01WC=" + wordAt(0) + "," + wordClassAt(1));
      maybeAddFeat("12WC=" + wordAt(1) + "," + wordClassAt(2));
      feats.toArray;
    });
  }
}

object NerExample {
  
  def shapeFor(word: String) = {
    val result = new StringBuilder(word.length);
    var i = 0;
    while (i < word.length) {
      val c = word(i);
      val x = if (c.isLetter && c.isUpper) 'X' else if (c.isLetter) 'x' else if (c.isDigit) 'd' else c;
      if (result.length > 1 && (result.last == x) && result(result.length - 2) == x) {
        result += 'e'
      } else if (result.length > 1 && result.last == 'e' && result(result.length - 2) == x) {
        () // nothing
      } else {
        result += x;
      }
      i += 1;
    }
    result.toString
  }
    
  def classFor(word: String) = {
    val sb = new StringBuilder;
    val wlen = word.length();
    val numCaps = (word: Seq[Char]).count(_.isUpper);
    val hasDigit = word.exists(_.isDigit);
    val hasDash = word.contains('-');
    val hasLower = numCaps < wlen;
    val ch0 = word.charAt(0);
    val lowered = word.toLowerCase();
    if (Character.isUpperCase(ch0) || Character.isTitleCase(ch0)) {
      if (numCaps == 1) {
        sb.append("-INITC");
      } else {
        sb.append("-CAPS");
      }
    } else if (!Character.isLetter(ch0) && numCaps > 0) {
      sb.append("-CAPS");
    } else if (hasLower) {
      sb.append("-LC");
    }

    if (hasDigit) {
      sb.append("-NUM");
    }
    if (hasDash) {
      sb.append("-DASH");
    }
    if (lowered.endsWith("s") && wlen >= 3) {
      // here length 3, so you don't miss out on ones like 80s
      val ch2 = lowered.charAt(wlen - 2);
      // not -ess suffixes or greek/latin -us, -is
      if (ch2 != 's' && ch2 != 'i' && ch2 != 'u') {
        sb.append("-s");
      }
    } else if (word.length() >= 5 && !hasDash && !(hasDigit && numCaps > 0)) {
      if (lowered.endsWith("ed")) {
        sb.append("-ed");
      } else if (lowered.endsWith("ing")) {
        sb.append("-ing");
      } else if (lowered.endsWith("ion")) {
        sb.append("-ion");
      } else if (lowered.endsWith("er")) {
        sb.append("-er");
      } else if (lowered.endsWith("est")) {
        sb.append("-est");
      } else if (lowered.endsWith("ly")) {
        sb.append("-ly");
      } else if (lowered.endsWith("ity")) {
        sb.append("-ity");
      } else if (lowered.endsWith("y")) {
        sb.append("-y");
      } else if (lowered.endsWith("al")) {
        sb.append("-al");
      }
    }
    sb.toString;
  }
}