package edu.berkeley.nlp.coref

import java.io.PrintWriter
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.coref.preprocess.PreprocessingDriver
import edu.berkeley.nlp.futile.syntax.Tree
import edu.berkeley.nlp.coref.preprocess.Reprocessor
import scala.collection.mutable.HashSet
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Logger

object ConllDocWriter {

  def writeDoc(writer: PrintWriter, conllDoc: ConllDoc, clustering: OrderedClusteringBound) {
//    writeDocIncompleteConll(writer, conllDoc.docID, conllDoc.docPartNo, conllDoc.words, conllDoc.pos, conllDoc.trees.map(_.constTree), conllDoc.speakers, conllDoc.nerChunks, convertOrderedClusteringBoundToChunks(clustering, conllDoc.words.size));  
    val corefBits = getCorefBits(conllDoc.words, convertOrderedClusteringBoundToChunks(clustering, conllDoc.words.size));
    val numZeroesToAddToPartNo = 3 - conllDoc.docPartNo.toString.size;
    writer.println("#begin document (" + conllDoc.docID + "); part " + ("0" * numZeroesToAddToPartNo) + conllDoc.docPartNo);
    for (sentIdx <- 0 until conllDoc.rawText.size) {
      val sent = conllDoc.rawText(sentIdx);
      for (tokenIdx <- 0 until sent.size) {
        val line = conllDoc.rawText(sentIdx)(tokenIdx);
        val lineNoCoref = line.substring(0, Math.max(line.lastIndexOf("\t"), line.lastIndexOf(" ")) + 1);
//        writer.println(lineNoCoref + corefBits(sentIdx)(tokenIdx));
        writer.println(lineNoCoref.replaceAll("\\s+", "\t") + corefBits(sentIdx)(tokenIdx));
      }
      writer.println();
    }
    writer.println("#end document");
  }
  
  // Doesn't write predicate-argument structures, senses, or lemmas (but we don't use these).
  def writeIncompleteConllDoc(writer: PrintWriter,
                              docName: String,
                              partNo: Int,
                              words: Seq[Seq[String]],
                              pos: Seq[Seq[String]],
                              parses: Seq[Tree[String]],
                              speakers: Seq[Seq[String]],
                              nerChunks: Seq[Seq[Chunk[String]]],
                              corefChunks: Seq[Seq[Chunk[Int]]]) {
    val numZeroesToAddToPartNo = 3 - partNo.toString.size;
    val corefBits = getCorefBits(words, corefChunks);
    val parseBits = parses.map(tree => PreprocessingDriver.computeParseBits(Reprocessor.convertFromFutileTree(tree)));
    val nerBits = getNerBits(words, nerChunks);
    writer.println("#begin document (" + docName + "); part " + ("0" * numZeroesToAddToPartNo) + partNo);
    for (sentIdx <- 0 until words.size) {
      val sent = words(sentIdx);
      for (i <- 0 until sent.size) {
        writer.println(docName + "\t" + partNo + "\t" + i + "\t" + words(sentIdx)(i) + "\t" + pos(sentIdx)(i) + "\t" + parseBits(sentIdx)(i) +
          "\t-\t-\t-\t" + speakers(sentIdx)(i) + "\t" + nerBits(sentIdx)(i) + "\t" + corefBits(sentIdx)(i));
      }
      writer.println();
    }
    writer.println("#end document");
  }
  
  private def convertOrderedClusteringBoundToChunks(clustering: OrderedClusteringBound, numSentences: Int): Seq[Seq[Chunk[Int]]] = {
    val chunksPerSentence = Array.tabulate(numSentences)(i => new ArrayBuffer[Chunk[Int]]());
    for (i <- 0 until clustering.ments.size) {
      val ment = clustering.ments(i);
      chunksPerSentence(ment.sentIdx) += new Chunk(ment.startIdx, ment.endIdx, clustering.clustering.getClusterIdx(i));
    }
    chunksPerSentence;
  }
  
  private def getNerBits(words: Seq[Seq[String]], nerChunks: Seq[Seq[Chunk[String]]]): Seq[Seq[String]] = {
    for (sentIdx <- 0 until words.size) yield {
      val chunkStarts = new HashMap[Int,String];
      val chunkEnds = new HashSet[Int];
      Logger.logss("NER CHUNKS: " + nerChunks);
      for (chunk <- nerChunks(sentIdx)) {
        chunkStarts.put(chunk.start, chunk.label);
        chunkEnds += chunk.end - 1;
      }
      for (tokenIdx <- 0 until words(sentIdx).size) yield {
        if (chunkStarts.contains(tokenIdx) && chunkEnds.contains(tokenIdx)) {
          "(" + chunkStarts.get(tokenIdx).getOrElse("") + ")";
        } else if (chunkStarts.contains(tokenIdx)) {
          "(" + chunkStarts.get(tokenIdx).getOrElse("") + "*";
        } else if (chunkEnds.contains(tokenIdx)) {
          "*)";
        } else {
          "*";
        }
      }
    }
  }
  
  private def getCorefBits(words: Seq[Seq[String]], corefChunks: Seq[Seq[Chunk[Int]]]): Seq[Seq[String]] = {
    for (sentIdx <- 0 until words.size) yield {
      val mentionStarts = new HashMap[Int,ArrayBuffer[Int]];
      val mentionEnds = new HashMap[Int,ArrayBuffer[Int]];
      val mentionStartEnds = new HashMap[Int,Int];
      val chunksThisSent = corefChunks(sentIdx);
      for (chunk <- chunksThisSent) {
        val start = chunk.start;
        val end = chunk.end - 1;
        if (start == end) {
          mentionStartEnds.put(start, chunk.label);
        } else {
          if (!mentionStarts.contains(start)) {
            mentionStarts.put(start, new ArrayBuffer[Int]())
          }
          mentionStarts(start) += chunk.label;
          if (!mentionEnds.contains(end)) {
            mentionEnds.put(end, new ArrayBuffer[Int]())
          }
          mentionEnds(end) += chunk.label;
        }
      }
      for (tokenIdx <- 0 until words(sentIdx).size) yield {
        var corefBit = "";
        if (mentionStarts.contains(tokenIdx)) {
          for (start <- mentionStarts(tokenIdx)) {
            corefBit += "(" + start + "|";
          }
        }
        if (mentionStartEnds.contains(tokenIdx)) {
          corefBit += "(" + mentionStartEnds(tokenIdx) + ")|";
        }
        if (mentionEnds.contains(tokenIdx)) {
          for (end <- mentionEnds(tokenIdx)) {
            corefBit += end + ")|";
          }
        }
        if (corefBit.isEmpty) "-" else corefBit.dropRight(1);
      }
    }
  }
}