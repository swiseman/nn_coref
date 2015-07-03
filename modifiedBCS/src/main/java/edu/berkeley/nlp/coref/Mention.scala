package edu.berkeley.nlp.coref
import scala.collection.JavaConverters._
import edu.berkeley.nlp.coref.lang.CorefLanguagePack

// TODO: Extract an interface for ConllDoc so I don't have to keep the whole
// document around...but while I'm feature engineering it's useful to be able
// to put my hands on anything I want
class Mention(val rawDoc: ConllDoc,
              val mentIdx: Int,
              val sentIdx: Int,
              val startIdx: Int,
              val endIdx: Int,
              val headIdx: Int,
              val mentionType: MentionType,
              val nerString: String,
              val number: Number,
              val gender: Gender) {
  
  // Cache these computations since they happen for every feature...
//  val cachedRawPronConjStr = if (mentionType == MentionType.PRONOMINAL) headStringLc else mentionType.toString;
  private val cachedRawPronConjStr = if (mentionType.isClosedClass) headStringLc else mentionType.toString;
//  val cachedCanonicalPronConjStr = if (mentionType == MentionType.PRONOMINAL) {
  private val cachedCanonicalPronConjStr = if (mentionType.isClosedClass) {
    if (!PronounDictionary.canonicalize(headStringLc).equals("")) {
      PronounDictionary.canonicalize(headStringLc);
    } else {
      headStringLc;
    }
  } else {
    mentionType.toString();
  }
  private var cachedCanonicalOrCommonConjStr = "";

  def speaker = rawDoc.speakers(sentIdx)(headIdx);

  def headString = rawDoc.words(sentIdx)(headIdx);
  def headStringLc = rawDoc.words(sentIdx)(headIdx).toLowerCase;
  def words = rawDoc.words(sentIdx).slice(startIdx, endIdx); 
  def pos = rawDoc.pos(sentIdx).slice(startIdx, endIdx);
  def spanToString = rawDoc.words(sentIdx).slice(startIdx, endIdx).reduce(_ + " " + _);
  
  def accessWordOrPlaceholder(idx: Int) = {
    if (idx < 0) Mention.StartWordPlaceholder else if (idx >= rawDoc.words(sentIdx).size) Mention.EndWordPlaceholder else rawDoc.words(sentIdx)(idx);
  }
  
  def accessPosOrPlaceholder(idx: Int) = {
    if (idx < 0) Mention.StartPosPlaceholder else if (idx >= rawDoc.pos(sentIdx).size) Mention.EndPosPlaceholder else rawDoc.pos(sentIdx)(idx);
  }
  
  def contextWordOrPlaceholder(idx: Int) = accessWordOrPlaceholder(startIdx + idx);
  def contextPosOrPlaceholder(idx: Int) = accessPosOrPlaceholder(startIdx + idx);
  
  def governor = governorHelper(false);
  def governorPos = governorHelper(true);
  private def governorHelper(pos: Boolean) = {
    val parentIdx = rawDoc.trees(sentIdx).childParentDepMap(headIdx);
    if (parentIdx == -1) {
      "[ROOT]"
    } else {
      (if (headIdx < parentIdx) "L" else "R") + "-" + (if (pos) rawDoc.pos(sentIdx)(parentIdx) else rawDoc.words(sentIdx)(parentIdx));
    }
  }
  
//  private def wordsFromBaseIndexAndOffset(baseIdx: Int, offsets: Seq[Int]) = offsets.map(offset => accessWordOrPlaceholder(baseIdx + offset)).reduce(_ + " " + _)
//  private def possFromBaseIndexAndOffset(baseIdx: Int, offsets: Seq[Int]) = offsets.map(offset => accessPosOrPlaceholder(baseIdx + offset)).reduce(_ + " " + _)
//  
//  def wordsFromStart(offsets: Seq[Int]) = wordsFromBaseIndexAndOffset(startIdx, offsets);
//  def wordsFromHead(offsets: Seq[Int]) = wordsFromBaseIndexAndOffset(headIdx, offsets);
//  def wordsFromEnd(offsets: Seq[Int]) = wordsFromBaseIndexAndOffset(endIdx, offsets);
//  def possFromStart(offsets: Seq[Int]) = possFromBaseIndexAndOffset(startIdx, offsets);
//  def possFromHead(offsets: Seq[Int]) = possFromBaseIndexAndOffset(headIdx, offsets);
//  def possFromEnd(offsets: Seq[Int]) = possFromBaseIndexAndOffset(endIdx, offsets);
//  
//  def wordFromStart(offset: Int) = accessWordOrPlaceholder(startIdx + offset);
//  def wordFromHead(offset: Int) = accessWordOrPlaceholder(headIdx + offset);
//  def wordFromEnd(offset: Int) = accessWordOrPlaceholder(endIdx + offset);
//  def posFromStart(offset: Int) = accessPosOrPlaceholder(startIdx + offset);
//  def posFromHead(offset: Int) = accessPosOrPlaceholder(headIdx + offset);
//  def posFromEnd(offset: Int) = accessPosOrPlaceholder(endIdx + offset);
  
  // These are explicit rather than in terms of Seq[Int] for lower overhead during
  // feature computation.
//  def wordBigramFromStart(offset1: Int, offset2: Int) = accessWordOrPlaceholder(startIdx + offset1) + " " + accessWordOrPlaceholder(startIdx + offset2);
//  def wordBigramFromHead(offset1: Int, offset2: Int) = accessWordOrPlaceholder(headIdx + offset1) + " " + accessWordOrPlaceholder(headIdx + offset2);
//  def wordBigramFromEnd(offset1: Int, offset2: Int) = accessWordOrPlaceholder(endIdx + offset1) + " " + accessWordOrPlaceholder(endIdx + offset2);
//  def posBigramFromStart(offset1: Int, offset2: Int) = accessPosOrPlaceholder(startIdx + offset1) + " " + accessPosOrPlaceholder(startIdx + offset2);
//  def posBigramFromHead(offset1: Int, offset2: Int) = accessPosOrPlaceholder(headIdx + offset1) + " " + accessPosOrPlaceholder(headIdx + offset2);
//  def posBigramFromEnd(offset1: Int, offset2: Int) = accessPosOrPlaceholder(endIdx + offset1) + " " + accessPosOrPlaceholder(endIdx + offset2);
  
  def computeBasicConjunctionStr = mentionType.toString;
  def computeRawPronounsConjunctionStr = cachedRawPronConjStr;
  def computeCanonicalPronounsConjunctionStr = cachedCanonicalPronConjStr;
  def computeCanonicalOrCommonConjunctionStr(lexicalCountsBundle: LexicalCountsBundle) = {
    if (cachedCanonicalOrCommonConjStr == "") {
      cachedCanonicalOrCommonConjStr = if (mentionType.isClosedClass || lexicalCountsBundle.commonHeadWordCounts.getCount(headStringLc) < 500) {
        cachedCanonicalPronConjStr
      } else {
        headStringLc;
      }
    }
    cachedCanonicalOrCommonConjStr;
  }
  
  def iWi(other: Mention) = {
    sentIdx == other.sentIdx && ((other.startIdx <= this.startIdx && this.endIdx <= other.endIdx) ||
                                 (this.startIdx <= other.startIdx && other.endIdx <= this.endIdx));
  }
  
  def computeSyntacticUnigram: String = rawDoc.trees(sentIdx).computeSyntacticUnigram(headIdx);
  def computeSyntacticBigram: String = rawDoc.trees(sentIdx).computeSyntacticBigram(headIdx);
  def computeSyntacticPosition: String = rawDoc.trees(sentIdx).computeSyntacticPositionSimple(headIdx);
}

object Mention {
  
  val StartWordPlaceholder = "<s>";
  val EndWordPlaceholder = "</s>";
  val StartPosPlaceholder = "<S>";
  val EndPosPlaceholder = "</S>";
  
  def createMentionComputeProperties(rawDoc: ConllDoc,
                                     mentIdx: Int,
                                     sentIdx: Int,
                                     startIdx: Int,
                                     endIdx: Int,
                                     headIdx: Int,
                                     propertyComputer: MentionPropertyComputer,
                                     langPack: CorefLanguagePack): Mention = {
    // NER
    var nerString = "O";
    for (chunk <- rawDoc.nerChunks(sentIdx)) {
      if (chunk.start <= headIdx && headIdx < chunk.end) {
        nerString = chunk.label;
      }
    }
    // MENTION TYPE
    var mentionType = if (endIdx - startIdx == 1 && PronounDictionary.isDemonstrative(rawDoc.words(sentIdx)(headIdx))) {
      MentionType.DEMONSTRATIVE;
    } else if (endIdx - startIdx == 1 && (PronounDictionary.isPronLc(rawDoc.words(sentIdx)(headIdx).toLowerCase) || langPack.getPronominalTags.contains(rawDoc.pos(sentIdx)(headIdx)))) {
      MentionType.PRONOMINAL;
    } else if (nerString != "O" || langPack.getProperTags.contains(rawDoc.pos(sentIdx)(headIdx))) {
      MentionType.PROPER;
    } else {
      MentionType.NOMINAL;
    }
    // GENDER AND NUMBER
    var number: Number = Number.SINGULAR;
    var gender: Gender = Gender.MALE;
    if (mentionType == MentionType.PRONOMINAL) {
      val pronLc = rawDoc.words(sentIdx)(headIdx).toLowerCase;
      gender = if (PronounDictionary.malePronouns.contains(pronLc)) {
        Gender.MALE 
      } else if (PronounDictionary.femalePronouns.contains(pronLc)) {
        Gender.FEMALE
      } else if (PronounDictionary.neutralPronouns.contains(pronLc)) {
        Gender.NEUTRAL;
      } else {
        Gender.UNKNOWN;
      }
      number = if (PronounDictionary.singularPronouns.contains(pronLc)) {
        Number.SINGULAR
      } else if (PronounDictionary.pluralPronouns.contains(pronLc)) {
        Number.PLURAL;
      } else {
        Number.UNKNOWN;
      }
    } else {
      if (propertyComputer.ngComputer != null) {
        number = propertyComputer.ngComputer.computeNumber(rawDoc.words(sentIdx).slice(startIdx, endIdx), rawDoc.words(sentIdx)(headIdx));
        gender = if (nerString == "PERSON") {
          propertyComputer.ngComputer.computeGenderPerson(rawDoc.words(sentIdx).slice(startIdx, endIdx), headIdx - startIdx);
        } else {
          propertyComputer.ngComputer.computeGenderNonPerson(rawDoc.words(sentIdx).slice(startIdx, endIdx), rawDoc.words(sentIdx)(headIdx));
        }
      }
    }
    return new Mention(rawDoc, mentIdx, sentIdx, startIdx, endIdx, headIdx, mentionType, nerString, number, gender);
  }
}

