package edu.berkeley.nlp.coref
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.coref.lang.CorefLanguagePack
import edu.berkeley.nlp.coref.lang.Language
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.lang.EnglishCorefLanguagePack
import edu.berkeley.nlp.coref.lang.ChineseCorefLanguagePack
import edu.berkeley.nlp.coref.lang.ArabicCorefLanguagePack

case class ProtoMention(val sentIdx: Int, val startIdx: Int, val endIdx: Int, val headIdx: Int);

class CorefDocAssembler(val langPack: CorefLanguagePack,
                        val useGoldMentions: Boolean) {
  
  def createCorefDoc(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer): CorefDoc = {
    val (goldMentions, goldClustering) = extractGoldMentions(rawDoc, propertyComputer);
    if (goldMentions.size == 0) {
      Logger.logss("WARNING: no gold mentions on document " + rawDoc.printableDocName);
    }
    val predMentions = if (useGoldMentions) {
      goldMentions;
    } else {
      extractPredMentions(rawDoc, propertyComputer);
    }
    new CorefDoc(rawDoc, goldMentions, goldClustering, predMentions)
  }
  
  def extractGoldMentions(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer): (Seq[Mention], OrderedClustering) = {
    val goldProtoMentionsSorted = getGoldProtoMentionsSorted(rawDoc);
    val finalMentions = new ArrayBuffer[Mention]();
    val goldClusterLabels = new ArrayBuffer[Int]();
    for (sentProtoMents <- goldProtoMentionsSorted; protoMent <- sentProtoMents) {
      finalMentions += Mention.createMentionComputeProperties(rawDoc, finalMentions.size, protoMent.sentIdx, protoMent.startIdx, protoMent.endIdx, protoMent.headIdx, propertyComputer, langPack)
      val correspondingChunks = rawDoc.corefChunks(protoMent.sentIdx).filter(chunk => chunk.start == protoMent.startIdx && chunk.end == protoMent.endIdx);
      if (correspondingChunks.size != 1) {
        Logger.logss("WARNING: multiple gold coref chunks matching span");
        Logger.logss("Location: " + rawDoc.printableDocName + ", sentence " + protoMent.sentIdx + ": (" + protoMent.startIdx + ", " + protoMent.endIdx + ") " +
                     rawDoc.words(protoMent.sentIdx).slice(protoMent.startIdx, protoMent.endIdx));
      }
      require(correspondingChunks.size >= 1);
      goldClusterLabels += correspondingChunks.map(_.label).reduce(Math.min(_, _));
    }
    (finalMentions, OrderedClustering.createFromClusterIds(goldClusterLabels));
  }
  
  def extractPredMentions(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer): Seq[Mention] = {
    val protoMentionsSorted = getProtoMentionsSorted(rawDoc);
    val finalMentions = new ArrayBuffer[Mention]();
    for (sentProtoMents <- protoMentionsSorted; protoMent <- sentProtoMents) {
      finalMentions += Mention.createMentionComputeProperties(rawDoc, finalMentions.size, protoMent.sentIdx, protoMent.startIdx, protoMent.endIdx, protoMent.headIdx, propertyComputer, langPack)
    }
    finalMentions;
  }
  
  private def getGoldProtoMentionsSorted(rawDoc: ConllDoc): Seq[Seq[ProtoMention]] = {
    val goldProtoMentions = for (sentIdx <- 0 until rawDoc.corefChunks.size) yield {
       for (chunk <- rawDoc.corefChunks(sentIdx)) yield {
         val headIdx = rawDoc.trees(sentIdx).getSpanHead(chunk.start, chunk.end);
         new ProtoMention(sentIdx, chunk.start, chunk.end, headIdx);
       }
    }
    goldProtoMentions.map(sortProtoMentionsLinear(_));
  }
  
  private def getProtoMentionsSorted(rawDoc: ConllDoc): Seq[Seq[ProtoMention]] = {
    val mentionExtents = (0 until rawDoc.numSents).map(i => new HashSet[ProtoMention]);
    for (sentIdx <- 0 until rawDoc.numSents) {
      // Extract NE spans: filter out O, QUANTITY, CARDINAL, CHUNK
      // Throw out NE types which aren't mentions
      val filterNEsByType: Chunk[String] => Boolean = chunk => !(chunk.label == "O" || chunk.label == "QUANTITY" || chunk.label == "CARDINAL" || chunk.label == "PERCENT");
      // ENGLISH ONLY: Annoyingly, NE's often don't contain trailing 's, so add this manually
      val expandNEsToIncludePossessive: Chunk[String] => Chunk[String] = chunk => {
        val sent = rawDoc.words(sentIdx);
        if (chunk.end <= sent.size - 1 && sent(chunk.end) == "'s") new Chunk[String](chunk.start, chunk.end + 1, chunk.label) else chunk
      }
      val neProtoMentions = rawDoc.nerChunks(sentIdx).filter(filterNEsByType).map(expandNEsToIncludePossessive).
          map(chunk => new ProtoMention(sentIdx, chunk.start, chunk.end, rawDoc.trees(sentIdx).getSpanHead(chunk.start, chunk.end)));
      mentionExtents(sentIdx) ++= neProtoMentions
      // Extract NPs and PRPs *except* for those contained in NE chunks (the NE tagger seems more reliable than the parser)
      val filterSpanIfInNE: ((Int, Int, Int)) => Boolean = startEndHead => neProtoMentions.filter(ment => ment.startIdx <= startEndHead._1 && startEndHead._2 <= ment.endIdx).size == 0;
      val posAndConstituentsOfInterest = langPack.getMentionConstituentTypes ++ langPack.getPronominalTags;
      for (label <- posAndConstituentsOfInterest) {
        mentionExtents(sentIdx) ++= rawDoc.trees(sentIdx).getSpansAndHeadsOfType(label).filter(filterSpanIfInNE).map(span => new ProtoMention(sentIdx, span._1, span._2, span._3));
      }
    }
    // Now take maximal mentions with the same heads
    val filteredProtoMentionsSorted = (0 until rawDoc.numSents).map(i => new ArrayBuffer[ProtoMention]);
    for (sentIdx <- 0 until mentionExtents.size) {
      val protoMentionsByHead = mentionExtents(sentIdx).groupBy(_.headIdx);
      // Look from smallest head first
      for (head <- protoMentionsByHead.keys.toSeq.sorted) {
        // Find the biggest span containing this head
        var currentBiggest: ProtoMention = null;
        for (ment <- protoMentionsByHead(head)) {
          // Overlapping but neither is contained in the other
          if (currentBiggest != null && ((ment.startIdx < currentBiggest.startIdx && ment.endIdx < currentBiggest.endIdx) || (ment.startIdx > currentBiggest.startIdx && ment.endIdx > currentBiggest.endIdx))) {
            Logger.logss("WARNING: mentions with the same head but neither contains the other");
            Logger.logss("  " + rawDoc.words(sentIdx).slice(ment.startIdx, ment.endIdx) + ", head = " + rawDoc.words(sentIdx)(head));
            Logger.logss("  " + rawDoc.words(sentIdx).slice(currentBiggest.startIdx, currentBiggest.endIdx) + ", head = " + rawDoc.words(sentIdx)(head));
          }
          // This one is bigger
          if (currentBiggest == null || (ment.startIdx <= currentBiggest.startIdx && ment.endIdx >= currentBiggest.endIdx)) {
            currentBiggest = ment;
          }
        }
        filteredProtoMentionsSorted(sentIdx) += currentBiggest;
        // ENGLISH ONLY: don't remove appositives
        for (ment <- protoMentionsByHead(head)) {
          val isNotBiggest = ment.startIdx != currentBiggest.startIdx || ment.endIdx != currentBiggest.endIdx;
          val isAppositiveLike = ment.endIdx < rawDoc.pos(sentIdx).size && (rawDoc.pos(sentIdx)(ment.endIdx) == "," || rawDoc.pos(sentIdx)(ment.endIdx) == "CC");
          if (isNotBiggest && isAppositiveLike && Driver.includeAppositives) {
            filteredProtoMentionsSorted(sentIdx) += ment;
          }
        }
      }
    }
    filteredProtoMentionsSorted.map(sortProtoMentionsLinear(_));
  }
  
  private def sortProtoMentionsLinear(protoMentions: Seq[ProtoMention]): Seq[ProtoMention] = {
    protoMentions.sortBy(ment => (ment.sentIdx, ment.headIdx, ment.endIdx, ment.startIdx));
  }
}

object CorefDocAssembler {
  
  def apply(language: Language, useGoldMentions: Boolean) = {
    val langPack = language match {
      case Language.ENGLISH => new EnglishCorefLanguagePack();
      case Language.CHINESE => new ChineseCorefLanguagePack(); 
      case Language.ARABIC => new ArabicCorefLanguagePack();
      case _ => throw new RuntimeException("Unrecognized language");
    }
    new CorefDocAssembler(langPack, useGoldMentions);
  }
}
