package edu.harvard.nlp.moarcoref

import java.io.PrintWriter
import scala.collection.JavaConverters.mapAsScalaMapConverter
import scala.collection.immutable.TreeMap
import edu.berkeley.nlp.coref.CorefFeaturizerTrainer
import edu.berkeley.nlp.coref.CorefSystem
import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.coref.Mention
import edu.berkeley.nlp.coref.NumberGenderComputer
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizerJoint
import edu.berkeley.nlp.coref.sem.QueryCountsBundle
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.coref.Chunk
import edu.berkeley.nlp.coref.ConllDoc

object ContextWriter {
    
  def writeAllMentsInContext() {
    
    // read in google news words
    val vocab = AnimacyHelper.getWordsFromFile("GoogleNewsJustWords.txt", false);
    Logger.logss("read in vocab with " + vocab.size + " words");
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(MiniDriver.numberGenderDataPath);
    require(!MiniDriver.trainOnGold);

    var trainDocs = CorefSystem.loadCorefDocs(MiniDriver.trainPath, MiniDriver.trainSize, numberGenderComputer, MiniDriver.trainOnGold);
    var trainDocGraphsOrigOrder = trainDocs.map(new DocumentGraph(_, true));
    var trainDocGraphs = if (MiniDriver.randomizeTrain) new scala.util.Random(0).shuffle(trainDocGraphsOrigOrder.sortBy(_.corefDoc.rawDoc.printableDocName)) else trainDocGraphsOrigOrder;
    writeMentsInContext("trainCtx.txt", "trainMentOffsets.txt", trainDocGraphs, vocab);


    var devDocs = CorefSystem.loadCorefDocs(MiniDriver.devPath, MiniDriver.devSize, numberGenderComputer, MiniDriver.trainOnGold);
    var devDocGraphs = devDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    writeMentsInContext("devCtx.txt", "devMentOffsets.txt", devDocGraphs, vocab);
    
    var testDocs = CorefSystem.loadCorefDocs(MiniDriver.testPath, MiniDriver.testSize, numberGenderComputer, MiniDriver.trainOnGold);
    var testDocGraphs = testDocs.map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);
    writeMentsInContext("testCtx.txt", "testMentOffsets.txt", testDocGraphs, vocab);    
  } 


//  // this assumes all named entity strings will be inside mentions, so we only need to join things there
 def writeMentsInContext(ctxFiName:String, mentFiName:String, dgs: Seq[DocumentGraph], vocab: java.util.Set[String]){
    val cp = new PrintWriter(ctxFiName);
    val mp = new PrintWriter(mentFiName);
    for (dg <- dgs){
      val ments = dg.corefDoc.predMentions.toArray;
      // partition ments by sentence
      val mentsBySent = Array.fill[ArrayBuffer[Mention]](dg.corefDoc.rawDoc.numSents)(ArrayBuffer[Mention]());
      var m = 0;
      while (m < ments.length){
        mentsBySent(ments(m).sentIdx) += ments(m);
        m += 1;
      }
      // write (merged sentences) and mentions
      var s = 0;
      var tokensSoFar = 0;
      while (s < dg.corefDoc.rawDoc.numSents){
        if (s != 0){
          cp.print(" ");
        }
        // get new merged sentence
        val (newSent, idxMap) = mergedSentStuff(dg.corefDoc.rawDoc,s,dg.corefDoc.rawDoc.nerChunks(s),vocab);
        cp.print(newSent.mkString(" "));
        cp.print(" </s>"); 
        var nm = 0;
        while (nm < mentsBySent(s).size){
          if (s != 0 || nm != 0){
            mp.print(" ");
          }
          mp.print(tokensSoFar + idxMap(mentsBySent(s)(nm).startIdx));
          mp.print(":");
          mp.print(tokensSoFar + idxMap(mentsBySent(s)(nm).endIdx));
          nm += 1;
        }
        tokensSoFar += newSent.length+1; // add 1 for </s>
        s += 1;
      }
      cp.println();
      mp.println();
    }
    cp.flush();
    cp.close();
    mp.flush();
    mp.close();
  }  

   def mergedSentStuff(doc:ConllDoc, sentIdx:Int, sentNERChunks:Seq[Chunk[String]], vocab:java.util.Set[String]):(Array[String], Array[Int]) = {
    val origSent = doc.words(sentIdx).toArray;
    // map digits to #
    var i = 0
    while (i < origSent.length){
      origSent(i) = origSent(i).map(c => if (c.isDigit) '#' else c);
      i += 1;
    }
    //val origCopy = origSent.clone;
    // merge things left
    for (chunk <- sentNERChunks) {
        assert(origSent(chunk.start) != null); // we assume chunks do not overlap
//        if (chunk.label == "QUANTITY" || chunk.label == "CARDINAL"){
//          origSent(chunk.start) = "NUM";
//          var i = chunk.start+1;
//          while (i < chunk.end){
//            origSent(i) = null;
//            i += 1;
//          }
//        } else 
//        if (chunk.label == "PERCENT"){
//          origSent(chunk.start) = "%";
//          var i = chunk.start+1;
//          while (i < chunk.end){
//            origSent(i) = null;
//            i += 1;
//          }
//        } else if (chunk.label == "MONEY"){
//          origSent(chunk.start) = "$"; // this may be too coarse
//          var i = chunk.start+1;
//          while (i < chunk.end){
//            origSent(i) = null;
//            i += 1;
//          }
//          // otherwise we'll just take it if google has an embedding for it
//        } //else if (chunk.label != "O" && chunk.label != "DATE" && chunk.label != "TIME" && chunk.label != "ORDINAL"
          //          && chunk.label != "WORK_OF_ART" && chunk.label != "EVENT" && chunk.label != "FAC" && chunk.label != "LAW") {
//        else {
          val candidateNE = doc.words(sentIdx).slice(chunk.start,chunk.end).reduce(_ + "_" + _);
          if (vocab.contains(candidateNE)) {
            origSent(chunk.start) = candidateNE;          
            var i = chunk.start+1;
            while (i < chunk.end){
              origSent(i) = null;
              i += 1;
            }            
          }
//        }        
    }
    // now map original indices to new indices
    val idxMap = Array.fill(origSent.length+1)(0); //catch index after sentence
    var currIdx = 0;
    var j = 0;
    while (j < origSent.length){
      idxMap(j) = currIdx;
      if (origSent(j) != null){
        currIdx += 1;
      }
      j += 1;
    }
    idxMap(j) = currIdx; // index after sentence
    val newSent = origSent.filter(_ != null);
    return (newSent, idxMap);
  }
 
//
//  // this assumes all named entity strings will be inside mentions, so we only need to join things there
//  def writeMentsInContext2(outFiName:String, dgs: Seq[DocumentGraph]){
//    val printer = new PrintWriter(outFiName);
//    for (dg <- dgs){
//      val ments = dg.corefDoc.predMentions.toArray;
//      val moarMents = ments.map(x => MoarMention.createMoarMention(x, dg.corefDoc.rawDoc.nerChunks(x.sentIdx)));
//      var m = 0;
//      var lastMent:Mention = null;
//      while (m < ments.length){
//        val currMent = ments(m);
//        var i = if (lastMent == null) 0 else lastMent.sentIdx;
//        val iStart = i;
//        while (i <= currMent.sentIdx){
//          var j = if (lastMent == null || i > iStart) 0 else lastMent.endIdx;
//          val ub = if (i == currMent.sentIdx) currMent.startIdx else dg.corefDoc.rawDoc.words(i).length;
//          if (j == 0){
//            printer.print("<s>");
//          }
//          while (j < ub){
//            printer.print(" ");
//            printer.print(dg.corefDoc.rawDoc.words(i)(j));
//            j += 1;
//          }
//          i += 1;
//        }
//        // get any enclosed mentions (but also current one)
//        var startIdxs = ArrayBuffer[Int]();
//        var endIdxs = ArrayBuffer[Int]();
//        var nextStart = currMent.startIdx;
//        while (m < ments.length && ments(m).startIdx < currMent.endIdx){
//          startIdxs += ments(m).startIdx;
//          endIdxs += ments(m).endIdx;
//          m += 1;
//        }
//        
//        // print mention
//        writeMent2(currMent, dg, startIdxs, endIdxs, printer);
//        lastMent = currMent;
//      }
//      printer.println();
//    }
//    printer.flush();
//    printer.close();
//  }  
//  
//  // this is stupid and not very efficient, but i don't want to mess with the whole interface etc
//  // assumes that ment is the highest mention (encapsulates most stuff..)
//  def writeMent2(ment:Mention, dg:DocumentGraph, startIdxs:ArrayBuffer[Int], endIdxs:ArrayBuffer[Int], printer:PrintWriter){
//    assert(isSorted(startIdxs));
//    assert(isSorted(endIdxs));
//    
//    // merge NE things
//    var idxs = ArrayBuffer[Int]();
//    // find chunks that overlap the mention
//    for (chunk <- dg.corefDoc.rawDoc.nerChunks(ment.sentIdx)) {
//      if (chunk.start <= ment.startIdx && chunk.end <= ment.endIdx) {
//        if (!chunk.label.equals("O")){
//          idxs += chunk.start;
//          idxs += chunk.end;
//        }
//      }
//    }
//    // now print stuff
//    printer.print("[");
//    var i = ment.startIdx;
//    var nextNEIdx = 0;
//    var nextStartIdx = 0;
//    var nextEndIdx = 0;
//    while (i < ment.endIdx){
//      
//      if (idxs.size > nextNEIdx && idxs(nextNEIdx) == i){
//        var j = i;
//        while (j < idxs(nextNEIdx+1)){
//          if (i > ment.startIdx && j == i){
//            printer.print(" ");
//          }
//          if (j > i){
//            printer.print("_");
//          }
//          if (nextStartIdx < startIdxs.size && startIdxs(nextStartIdx) == j){
//            printer.print("[");
//            nextStartIdx += 1;
//          }
//          printer.print(dg.corefDoc.rawDoc.words(ment.sentIdx)(j));
//          if (nextEndIdx < endIdxs.size && endIdxs(nextEndIdx) == j){
//            printer.print("]");
//            nextEndIdx += 1;
//          }
//          j += 1;
//        }
//        i = idxs(nextNEIdx+1);
//        nextNEIdx = nextNEIdx+2;
//      } else {
//        if (i > ment.startIdx){
//          printer.print(" ");
//        }
//        if (nextStartIdx < startIdxs.size && startIdxs(nextStartIdx) == i){
//          printer.print("[");
//          nextStartIdx += 1;
//        }        
//        printer.print(dg.corefDoc.rawDoc.words(ment.sentIdx)(i));
//        if (nextEndIdx < endIdxs.size && endIdxs(nextEndIdx) == i){
//          printer.print("]");
//          nextEndIdx += 1;
//        }        
//        i += 1;
//      }  
//    }
//    printer.print("]");
//  }
//  
//  // this is stupid and not very efficient, but i don't want to mess with the whole interface etc
//  def writeMent(ment:Mention, dg:DocumentGraph, printer:PrintWriter){
//    // merge NE things
//    var idxs = ArrayBuffer[Int]();
//    // find chunks that overlap the mention
//    for (chunk <- dg.corefDoc.rawDoc.nerChunks(ment.sentIdx)) {
//      if (chunk.start <= ment.startIdx && ment.endIdx < chunk.end) {
//        if (!chunk.label.equals("O")){
//          idxs += chunk.start;
//          idxs += chunk.end;
//        }
//      }
//    }
//    // now print stuff
//    printer.print("[");
//    var i = ment.startIdx;
//    var nextNEIdx = 0;
//    while (i < ment.endIdx){
//      if (idxs.size > nextNEIdx && idxs(nextNEIdx) == i){
//        var j = i;
//        while (j < idxs(nextNEIdx+1)){
//          if (i > ment.startIdx && j == i){
//            printer.print(" ");
//          }
//          if (j > i){
//            printer.print("_");
//          }
//          printer.print(dg.corefDoc.rawDoc.words(ment.sentIdx)(j));
//          j += 1;
//        }
//        i = idxs(nextNEIdx+1);
//        nextNEIdx = nextNEIdx+2;
//      } else {
//        if (i > ment.startIdx){
//          printer.print(" ");
//        }
//        printer.print(dg.corefDoc.rawDoc.words(ment.sentIdx)(i));
//        i += 1;
//      }  
//    }
//    printer.print("]");
//  }
//  
//  def isSorted(a:ArrayBuffer[Int]):Boolean = {
//    var i = 1;
//    while (i < a.size){
//      if (a(i) < a(i-1)){
//        return false;
//      }
//      i += 1;
//    }
//    return true;
//  }
//  
}