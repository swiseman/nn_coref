package edu.berkeley.nlp.coref
import java.util.IdentityHashMap

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.JavaConverters.bufferAsJavaListConverter
import scala.collection.JavaConverters.seqAsJavaListConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import edu.berkeley.nlp.coref.lang.Language
import edu.berkeley.nlp.coref.lang.ModArabicHeadFinder
import edu.berkeley.nlp.coref.lang.ModCollinsHeadFinder
import edu.berkeley.nlp.futile.ling.BikelChineseHeadFinder
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import edu.berkeley.nlp.futile.syntax.Tree
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.IOUtils

class ConllDocReader(val lang: Language) {
  
  val headFinder = lang match {
    case Language.ENGLISH => new ModCollinsHeadFinder();
    case Language.CHINESE => new BikelChineseHeadFinder();
    case Language.ARABIC => new ModArabicHeadFinder();
    case _ => throw new RuntimeException("Bad language, no head finder for " + lang);
  }
  
  def readConllDocs(fileName: String): Seq[ConllDoc] = {
    val conllDocs = new ArrayBuffer[ConllDoc];
    val allLines = IOUtils.readLinesHard(fileName).asScala;
    var docBySentencesByLines = new ArrayBuffer[ArrayBuffer[String]];
    var docID = "";
    var docPartNo = -1;
    // Split documents up into parts and sentences
    for (i <- 0 until allLines.size) {
      val line = allLines(i);
      if (line.startsWith("#begin document")) {
        val thisLineDocID = line.substring(allLines(i).indexOf("(") + 1, allLines(i).indexOf(")"));
        val thisLinePartNo = line.substring(line.indexOf("part ") + 5).trim.toInt;
        if (docID == "") {
          docID = thisLineDocID;
          docPartNo = thisLinePartNo;
        } else {
          // We just changed docIDs; flush the document
          conllDocs += assembleConllDoc(docBySentencesByLines, docID, docPartNo);
          docBySentencesByLines = new ArrayBuffer[ArrayBuffer[String]];
          docID = thisLineDocID;
          docPartNo = thisLinePartNo;
          docID = thisLineDocID;
        }
        docBySentencesByLines += new ArrayBuffer[String]();
      } else if (line.startsWith("#end document")) {
        // Do nothing
      } else if (line.trim.isEmpty) {
        docBySentencesByLines += new ArrayBuffer[String]();
      } else {
        docBySentencesByLines.last += line;
      }
    }
    conllDocs += assembleConllDoc(docBySentencesByLines, docID, docPartNo);
    conllDocs;
  }
  
  def assembleConllDoc(docBySentencesByLines: ArrayBuffer[ArrayBuffer[String]],
                       docID: String,
                       docPartNo: Int) = {
    // Filter any empty sentences that snuck in there
    var docBySentencesByLinesFixed = docBySentencesByLines.filter(!_.isEmpty);
    // Filter out any sentences that are too short
    docBySentencesByLinesFixed = docBySentencesByLinesFixed.map(sentence => {
      val badSentence = sentence.map(_.split("\\s+").size).reduce(Math.min(_, _)) < 12;
      if (badSentence) {
        Logger.logss("WARNING: Bad sentence, too few fields:\n" + sentence.reduce(_ + "\n" + _));
        if (sentence(0).startsWith("tc/ch/00/ch_0011") && sentence.size == 1) {
          val replacement = "tc/ch/00/ch_0011   2   0    fillerword    WRB   (TOP(FRAG*))  -   -   -    B  *   -"
          Logger.logss("Salvaging this sentence, replacing it with:\n" + replacement);
          ArrayBuffer(replacement);
        } else if (sentence(0).startsWith("tc/ch/00/ch_0021") && sentence.size == 1) {
          val replacement = "tc/ch/00/ch_0021   2   0    fillerword    ADD   (TOP(NP*))  -   -   -   -   *   -"
          Logger.logss("Salvaging this sentence, replacing it with:\n" + replacement);
          ArrayBuffer(replacement);
        } else {
          throw new RuntimeException("This sentence wasn't one of the CoNLL 2012 sentences we hardcoded in; you need to manually fix it");
        }
      } else {
        sentence;
      }
    });
    // Parse the individual sentences
    for (i <- 0 until docBySentencesByLinesFixed.size) {
      // Shouldn't have empty sentences
      require(!docBySentencesByLinesFixed(i).isEmpty, docBySentencesByLinesFixed.map(_.size.toString).reduce(_ + " " + _));
      for (j <- 0 until docBySentencesByLinesFixed(i).size) {
        // Shouldn't have empty lines
        require(!docBySentencesByLinesFixed(i)(j).trim.isEmpty);
      }
    }
    val docFields = docBySentencesByLinesFixed.map(_.map(_.split("\\s+")));
    val wordss = docFields.map(_.map(_(3)));
    val poss = docFields.map(_.map(_(4)));
    val parseBitss = docFields.map(_.map(_(5)));
    val speakerss = docFields.map(_.map(_(9)));
    val nerBitss = docFields.map(_.map(_(10)));
    val corefBitss = docFields.map(_.map(lineFields => lineFields(lineFields.size - 1)));
    new ConllDoc(docID,
                 docPartNo,
                 wordss,
                 poss,
                 (0 until wordss.size).map(i => assembleTree(wordss(i), poss(i), parseBitss(i))),
                 (if (Driver.useNer) nerBitss.map(assembleNerChunks(_)) else nerBitss.map(bits => Seq[Chunk[String]]())),
                 corefBitss.map(assembleCorefChunks(_)),
                 speakerss,
                 docBySentencesByLinesFixed)
  }
  
  def assembleTree(words: Seq[String], pos: Seq[String], parseBits: Seq[String]): DepConstTree = {
    var finalTree: Tree[String] = null;
    val stack = new ArrayBuffer[String];
    // When a constituent is closed, the guy on top of the stack will become
    // his parent. Build Trees as we go and register them with their parents so
    // that when we close the parents, their children are already all there.
    val childrenMap = new IdentityHashMap[String, ArrayBuffer[Tree[String]]];
    for (i <- 0 until parseBits.size) {
      require(parseBits(i).indexOf("*") != -1, parseBits(i) + " " + parseBits + "\n" + words);
      val openBit = parseBits(i).substring(0, parseBits(i).indexOf("*"));
      val closeBit = parseBits(i).substring(parseBits(i).indexOf("*") + 1);
      // Add to the end of the stack
      for (constituentType <- openBit.split("\\(").drop(1)) {
        // Make a new String explicitly so the IdentityHashMap works
        val constituentTypeStr = new String(constituentType);
        stack += constituentTypeStr;
        childrenMap.put(stack.last, new ArrayBuffer[Tree[String]]());
      }
      // Add the POS and word, which aren't specified in the parse bit but do need
      // to be in the Tree object
      val preterminalAndLeaf = new Tree[String](pos(i), IndexedSeq(new Tree[String](words(i))).asJava);
      childrenMap.get(stack.last) += preterminalAndLeaf;
      // Remove from the end of the stack
      var latestSubtree: Tree[String] = null;
      for (i <- 0 until closeBit.size) {
        require(closeBit(i) == ')');
        val constituentType = stack.last;
        stack.remove(stack.size - 1);
        latestSubtree = new Tree[String](constituentType, childrenMap.get(constituentType).asJava);
        if (!stack.isEmpty) {
          childrenMap.get(stack.last) += latestSubtree;
        }
      }
      if (stack.isEmpty) {
        finalTree = latestSubtree;
      }
    }
    require(finalTree != null, stack);
    // In Arabic, roots appear to be unlabeled sometimes, so fix this
    if (finalTree.getLabel() == "") {
      finalTree = new Tree[String]("ROOT", finalTree.getChildren);
    }
    val childParentMap = DepConstTree.extractDependencyStructure(finalTree, headFinder);
    val depTree = new DepConstTree(finalTree, pos, words, childParentMap);
    depTree;
  }
  
  def assembleNerChunks(nerBits: Seq[String]) = {
    val nerChunks = new ArrayBuffer[Chunk[String]]();
    var currStartIdx = -1;
    var currType = "";
    for (i <- 0 until nerBits.size) {
      val bit = nerBits(i);
      if (bit.contains("(")) {
        if (bit.contains(")")) {
          nerChunks += new Chunk[String](i, i+1, bit.substring(bit.indexOf("(") + 1, bit.indexOf(")")));
        } else {
          currStartIdx = i;
          currType = bit.substring(bit.indexOf("(") + 1, bit.indexOf("*"));
        }
      } else if (bit.contains(")")) {
        require(currStartIdx != -1, "Bad coref bits: " + nerBits);
        nerChunks += new Chunk[String](currStartIdx, i+1, currType);
        currStartIdx = -1;
        currType = "";
      }
    }
    nerChunks;
  }
  
  def assembleCorefChunks(corefBits: Seq[String]) = {
    val corefChunks = new ArrayBuffer[Chunk[Int]];
    // For each cluster, keep a stack of the indices where that cluster's guys started
    // (We need a stack because very occasionally there are nested coreferent mentions, often
    // in conversational text with disfluencies.)
    val exposedChunkStartIndices = new HashMap[Int,ArrayBuffer[Int]];
    for (i <- 0 until corefBits.size) {
      val bit = corefBits(i);
      if (bit != "-") {
        val parts = bit.split("\\|");
        for (part <- parts) {
          if (part.contains("(") && part.contains(")")) {
            corefChunks += new Chunk[Int](i, i+1, part.substring(part.indexOf("(") + 1, part.indexOf(")")).toInt);
          } else if (part.contains("(")) {
            val clusterIndex = part.substring(part.indexOf("(") + 1).toInt;
            if (!exposedChunkStartIndices.contains(clusterIndex)) {
              exposedChunkStartIndices.put(clusterIndex, new ArrayBuffer[Int]);
            }
            exposedChunkStartIndices(clusterIndex) += i;
          } else if (part.contains(")")) {
            val clusterIndex = part.substring(0, part.indexOf(")")).toInt;
            require(exposedChunkStartIndices.contains(clusterIndex), "Bad coref bit sequence: " + corefBits);
            val chunkStartIndexStack = exposedChunkStartIndices(clusterIndex);
            require(chunkStartIndexStack.size >= 1, "Bad coref bit sequence: " + corefBits);
            val startIndex = chunkStartIndexStack.remove(chunkStartIndexStack.size - 1);
            corefChunks += new Chunk[Int](startIndex, i+1, clusterIndex);
          } else {
            throw new RuntimeException("Bad part: " + part);
          }
        }
      }
    }
    // In wsj_0990 sentence 9 there are some chunks which have the same span but
    // different labels...filter these out and take the one with the smallest label.
    val corefChunksNoDupes = corefChunks.filter(chunk1 => {
      // Check if any identical chunk has a lower label, reduce on this (true if any chunk does),
      // then filter out if this is true
      !corefChunks.map(chunk2 => chunk1.start == chunk2.start && chunk1.end == chunk2.end && chunk2.label < chunk1.label).reduce(_ || _);
    });
    corefChunksNoDupes;
  }
}

object ConllDocReader {
  
  def main(args: Array[String]) {
//    testDoc("clean-data/smtrain/phoenix_0001.v2_auto_conll");
    testDoc("clean-data/problematic_ones/a2e_0024.v2_auto_conll");
//    testDoc("clean-data/problematic_ones/wsj_0990.v2_auto_conll");
  }
  
  def testDoc(fileName: String) {
    val reader = new ConllDocReader(Language.ENGLISH);
    val conllDocs = reader.readConllDocs(fileName);
    conllDocs.foreach(conllDoc => {
      println(conllDoc.docID + " " + conllDoc.docPartNo)
      val words = conllDoc.words;
      require(words.size == conllDoc.pos.size);
      require(words.size == conllDoc.trees.size);
      require(words.size == conllDoc.nerChunks.size);
      require(words.size == conllDoc.corefChunks.size);
      require(words.size == conllDoc.speakers.size);
      require(words.size == conllDoc.rawText.size);
      for (i <- 0 until words.size) {
        println("==SENTENCE " + i + "==");
        require(words(i).size == conllDoc.pos(i).size);
        println((0 until words(i).size).map(j => conllDoc.pos(i)(j) + ":" + words(i)(j)).reduce(_ + " " + _));
        println(PennTreeRenderer.render(conllDoc.trees(i).constTree));
        println("NER: " + conllDoc.nerChunks(i));
        println("COREF: " + conllDoc.corefChunks(i));
        println("SPEAKERS: " + conllDoc.speakers(i));
      }
    });
  }
}
