package edu.harvard.nlp.moarcoref

import edu.berkeley.nlp.coref.DocumentGraph
import java.io.PrintWriter
import scala.collection.mutable.HashSet
import scala.collection.mutable.TreeSet

object TextPickler {

  // we'll write in the following fmt. each doc will be on its own line. the line will start with the number of mentions
  // then will be feats_j0| .. |feats_jj|feats_{j+1}0|..|feats_{j+1}{j+1} etc. 
  def writeFeats(docGraphs: Seq[DocumentGraph], fiName: String){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size){
        var j = 0;
        while (j <= i){
          pw.print('|');
          val feats = dg.cachedFeats(i)(j);
          var k = 0;
          while (k < feats.length){
            pw.print(feats(k));
            if (k < feats.length - 1){
              pw.print(' ');
            }
            k += 1;
          }
          j += 1;
        }
        i += 1;
      }
      pw.println();
    }
    pw.close();
  }

  def writeFeats(docGraphs: Seq[DocumentGraph], fiName: String, prunedIndices: TreeSet[(Int, Int, Int)]) {
    val pw = new PrintWriter(fiName);
    for ((dg, d) <- docGraphs.zipWithIndex) {
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size) {
        var j = 0;
        while (j <= i) {
          pw.print('|');
          if (!prunedIndices.contains((d, i, j))) {
            val feats = dg.cachedFeats(i)(j);
            var k = 0;
            while (k < feats.length) {
              pw.print(feats(k));
              if (k < feats.length - 1) {
                pw.print(' ');
              }
              k += 1;
            }
          }
          j += 1;
        }
        i += 1;
      }
      pw.println();
    }
    pw.close();
  }  

  // just writes anaphoric feats for each mention
  def writeAnaphFeats(docGraphs: Seq[DocumentGraph], fiName: String){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size){
        pw.print('|');
        val feats = dg.cachedFeats(i)(i).sorted;
        var k = 0;
        while (k < feats.length){
          pw.print(feats(k));
          if (k < feats.length - 1){
            pw.print(' ');
            }
          k += 1;
        }        
        i += 1;
      }
      pw.println();
    }
    pw.close();
  }  

  def writePWFeats(docGraphs: Seq[DocumentGraph], biasFeatIdx:Int, fiName: String){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size){
        var j = 0;
        while (j < i){
          pw.print('|');
          val feats = dg.cachedFeats(i)(j);
          var k = 0;
          while (k < feats.length){
            pw.print(feats(k));
            if (k < feats.length - 1){
              pw.print(' ');
            }
            k += 1;
          }
          j += 1;
        }
        // now just write one bias feature for non-anaphoric option
        pw.print('|');
        pw.print(biasFeatIdx); // don't really use biasFeat anymore, but it indicates total number of features
        i += 1;
      }
      pw.println();
    }
    pw.close();
  }  
  
  
  // format will be a sequence of clusters separated by '|'. this can be used during training
  // and also for loss fcns.
  def writePredOracleClusterings(docGraphs: Seq[DocumentGraph], fiName: String){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      for ((clust,clustIdx) <- dg.getOraclePredClustering.clusters.zipWithIndex){
        if (clustIdx > 0){
          pw.print('|');
        }
        val clustSize = clust.size;
        for ((ment,mentIdx) <- clust.zipWithIndex){
          pw.print(ment);
          if (mentIdx < clustSize - 1){
            pw.print(' ');
          }
        }
      }
      pw.println();
    }
    pw.close();
  }
  
  def writeMentHeads(docGraphs: Seq[DocumentGraph],fiName: String, lowercase:Boolean = false){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size){
        pw.print('|');
        if (lowercase){
           pw.print(dg.corefDoc.predMentions(i).headStringLc);
        } else {
          pw.print(dg.corefDoc.predMentions(i).headString);
        }
        i += 1;
      }
      pw.println();
    }
    pw.close();
  }   
  
  
 def writeFullMentandCtx(docGraphs: Seq[DocumentGraph],fiName: String, lowercase:Boolean = false){
    val pw = new PrintWriter(fiName);
    for (dg <- docGraphs){
      pw.print(dg.size);
      var i = 0;
      while (i < dg.size){
        pw.print('|');
        val ment = dg.corefDoc.predMentions(i);
        if (lowercase){
            pw.print(ment.contextWordOrPlaceholder(-1).toLowerCase() + " [");
            pw.print(ment.spanToString.toLowerCase() + "] ");
            pw.print(ment.contextWordOrPlaceholder(ment.words.size).toLowerCase);
        } else {
            pw.print(ment.contextWordOrPlaceholder(-1) + " [");
            pw.print(dg.corefDoc.predMentions(i).spanToString + "] ");
            pw.print(ment.contextWordOrPlaceholder(ment.words.size));
        }
        i += 1;
      }
      pw.println();
    }
    pw.close();
  } 
  
}
