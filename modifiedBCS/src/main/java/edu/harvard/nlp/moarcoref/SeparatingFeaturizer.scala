package edu.harvard.nlp.moarcoref
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.HashMap
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.coref.sem.QueryCountsBundle
import edu.berkeley.nlp.coref.preprocess.NerExample
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.coref.ConjType
import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.coref.PronounDictionary
import edu.berkeley.nlp.coref.Mention
import edu.berkeley.nlp.coref.MentionType

// A modified version of BCS's PairwiseIndexingFeaturizerJoint.scala

@SerialVersionUID(1L)
class SeparatingFeaturizer(val featureIndexer: Indexer[String], 
                                      val featsToUse: String, 
                                      val conjType: ConjType, 
                                      val lexicalCounts: MoarLexicalCountsBundle, 
                                      val queryCounts: QueryCountsBundle, 
                                      val anaphoricityMode: Boolean = true) extends PairwiseIndexingFeaturizer with Serializable {
  def getIndexer = featureIndexer;
  
  def getIndex(feature: String, addToFeaturizer: Boolean): Int = {
    if (!addToFeaturizer) {
      if (!featureIndexer.contains(feature)) {
          val idx = featureIndexer.getIndex(SeparatingFeaturizer.UnkFeatName);
          require(idx == 0);
          idx;
        } else {
        featureIndexer.getIndex(feature);
      }
    } else {
      featureIndexer.getIndex(feature);
    }
  }
  
  def getQueryCountsBundle = queryCounts;
  
  def featurizeIndex(docGraph: DocumentGraph, currMentIdx: Int, antecedentIdx: Int, addToFeaturizer: Boolean): Seq[Int] = {
    featurizeIndexStandard(docGraph, currMentIdx, antecedentIdx, addToFeaturizer);
  }
  
  // this is just copied from BCS; we only ever use ConjType == None
  private def addFeatureAndConjunctions(feats: ArrayBuffer[Int],
                                        featName: String,
                                        currMent: Mention,
                                        antecedentMent: Mention,
                                        isPairFeature: Boolean,
                                        addToFeaturizer: Boolean) {
    if (conjType == ConjType.NONE) {
      feats += getIndex(featName, addToFeaturizer);
    } else if (conjType == ConjType.TYPE || conjType == ConjType.TYPE_OR_RAW_PRON || conjType == ConjType.CANONICAL || conjType == ConjType.CANONICAL_OR_COMMON) {
      val currConjunction = "&Curr=" + {
        if (conjType == ConjType.TYPE) {
          currMent.computeBasicConjunctionStr;
        } else if (conjType == ConjType.TYPE_OR_RAW_PRON) {
          currMent.computeRawPronounsConjunctionStr;
        } else if (conjType == ConjType.CANONICAL) {
          currMent.computeCanonicalPronounsConjunctionStr 
        } else {
          currMent.computeCanonicalOrCommonConjunctionStr(lexicalCounts);
        }
      }
      feats += getIndex(featName, addToFeaturizer);
      val featAndCurrConjunction = featName + currConjunction;
      feats += getIndex(featAndCurrConjunction, addToFeaturizer);
      if (currMent != antecedentMent) {
        val prevConjunction = "&Prev=" + {
          if (conjType == ConjType.TYPE) {  
            antecedentMent.computeBasicConjunctionStr;
          } else if (conjType == ConjType.TYPE_OR_RAW_PRON) {
            antecedentMent.computeRawPronounsConjunctionStr;
          } else if (conjType == ConjType.CANONICAL) {
            antecedentMent.computeCanonicalPronounsConjunctionStr 
          } else {
            antecedentMent.computeCanonicalOrCommonConjunctionStr(lexicalCounts);
          }
        }
        feats += getIndex(featAndCurrConjunction + prevConjunction, addToFeaturizer);
      }
    } else if (conjType == ConjType.CANONICAL_NOPRONPRON) {
      val specialCase = currMent != antecedentMent && currMent.mentionType.isClosedClass() && antecedentMent.mentionType.isClosedClass()
      val currConjunction = "&Curr=" + (if (specialCase) currMent.computeBasicConjunctionStr else currMent.computeCanonicalPronounsConjunctionStr);
      feats += getIndex(featName, addToFeaturizer);
      val featAndCurrConjunction = featName + currConjunction;
      feats += getIndex(featAndCurrConjunction, addToFeaturizer);
      if (currMent != antecedentMent) {
        val prevConjunction = "&Prev=" + (if (specialCase) antecedentMent.computeBasicConjunctionStr else antecedentMent.computeCanonicalPronounsConjunctionStr);
        feats += getIndex(featAndCurrConjunction + prevConjunction, addToFeaturizer);
      }
    } else if (conjType == ConjType.CANONICAL_ONLY_PAIR_CONJ) {
      feats += getIndex(featName, addToFeaturizer);
      if (isPairFeature) {
        val currConjunction = "&Curr=" + currMent.computeCanonicalPronounsConjunctionStr
        val featAndCurrConjunction = featName + currConjunction;
        feats += getIndex(featAndCurrConjunction, addToFeaturizer);
        if (currMent != antecedentMent) {
          val prevConjunction = "&Prev=" + antecedentMent.computeCanonicalPronounsConjunctionStr;
          feats += getIndex(featAndCurrConjunction + prevConjunction, addToFeaturizer);
        }
      }
    } else {
      throw new RuntimeException("Conjunction type not implemented");
    }
  }
  
  
  def featurizeIndexStandard(docGraph: DocumentGraph, currMentIdx: Int, antecedentIdx: Int, addToFeaturizer: Boolean): Seq[Int] = {
    val currMent = docGraph.getMention(currMentIdx);
    val antecedentMent = docGraph.getMention(antecedentIdx);
    val feats = new ArrayBuffer[Int]();
    
    if (currMentIdx == 0){ // ignore the first mention b/c it's never anaphoric
      return feats;
    }
    
    def addFeatureShortcut = (featName: String) => {
      // Only used in CANONICAL_ONLY_PAIR, so only compute the truth value in this case
      val isPairFeature = conjType == ConjType.CANONICAL_ONLY_PAIR_CONJ && !(featName.startsWith("SN") || featName.startsWith("PrevMent"));
      addFeatureAndConjunctions(feats, featName, currMent, antecedentMent, isPairFeature, addToFeaturizer);
    }
    // Features on anaphoricity
    val mentType = currMent.mentionType;
    val startingNew = antecedentIdx == currMentIdx;
    
    // only looking for features on the current ment in anaphoricityMode, and only pairwise features in non anaphoricityMode
    if (anaphoricityMode && !startingNew){ 
      return feats;
    } else if (!anaphoricityMode && startingNew){
      return feats;
    }
    
    if (startingNew && anaphoricityMode){
    
	    if (!featsToUse.contains("+nomentlen")) {
	      addFeatureShortcut("SNMentLen=" + currMent.words.size + "-SN=" + startingNew);
	    }

	    if (!featsToUse.contains("+nolexanaph")) {
	      addFeatureShortcut("SNMentHead=" + fetchHeadWordOrPos(currMent) + "-SN=" + startingNew);
	    }
	    
	    if (!featsToUse.contains("+nolexfirstword")) {
	      addFeatureShortcut("SNMentFirst=" + fetchFirstWordOrPos(currMent) + "-SN=" + startingNew);
	    }

	    if (!featsToUse.contains("+nolexlastword")) {
	      addFeatureShortcut("SNMentLast=" + fetchLastWordOrPos(currMent) + "-SN=" + startingNew);
	    }

	    if (!featsToUse.contains("+nolexprecedingword")) {
	      addFeatureShortcut("SNMentPreceding=" + fetchPrecedingWordOrPos(currMent) + "-SN=" + startingNew);
	    }

	    if (!featsToUse.contains("+nolexfollowingword")) {
	      addFeatureShortcut("SNMentFollowing=" + fetchFollowingWordOrPos(currMent) + "-SN=" + startingNew);
	    }

	    if (featsToUse.contains("FINAL")) {
	      addFeatureShortcut("SNSynPos=" + currMent.computeSyntacticUnigram);
	      addFeatureShortcut("SNSynPos=" + currMent.computeSyntacticBigram);
	    }	    
	    if (featsToUse.contains("MOARANAPH")) {
	      addFeatureShortcut("SNGovernor=" + fetchGovernorWordOrPos(currMent) + "-SN=" + startingNew);
	    }
	    if (featsToUse.contains("MOARANAPH")) {
	      addFeatureShortcut("SNSentMentIdx=" + computeSentMentIdx(docGraph, currMent) + "-SN=" + startingNew);
	    }
	    
	    // some of these features inspired by Recasens et al. (2013)
	    if (featsToUse.contains("MOARANAPH")){
	      addFeatureShortcut("SNNerStr=" + currMent.nerString + "-SN=" + startingNew);
	      addFeatureShortcut("SNAnimacy=" + AnimacyHelper.getAnimacy(currMent) + "-SN=" + startingNew);
	      if (currMent.mentionType == MentionType.PRONOMINAL){
	        addFeatureShortcut("SNGend=" + computePronGender(currMent) + "-SN=" + startingNew);
	      } else {
	        addFeatureShortcut("SNGend=" + currMent.gender + "-SN=" + startingNew);
	      }
	      if (currMent.mentionType == MentionType.PRONOMINAL){
	        addFeatureShortcut("SNNumber=" + computePronNumber(currMent) + "-SN=" + startingNew);
	      } else {
	        addFeatureShortcut("SNNumber=" + currMent.number + "-SN=" + startingNew);
	      }	      
	      addFeatureShortcut("SNPerson=" + computePronPerson(currMent) + "-SN=" + startingNew);
	    }
	    
	    // add raw pronoun canonical conjunction string if we're separating, since we don't do conjunctions
	    // Note "canonical pronoun conjunction string" is just what BCS calls it; we don't actually conjoin it with anything
	    addFeatureShortcut("CurrPronConjStr=" + currMent.computeCanonicalPronounsConjunctionStr + "-SN=" + startingNew);  
    }
        
    // Features just on the antecedent
    if (!startingNew && !anaphoricityMode) {

      if (!featsToUse.contains("+nomentlen")) {
        addFeatureShortcut("PrevMentLen=" + antecedentMent.words.size);
      }

      if (!featsToUse.contains("+nolexanaph")) {
        addFeatureShortcut("PrevMentHead=" + fetchHeadWordOrPos(antecedentMent));
      }

      if (!featsToUse.contains("+nolexfirstword")) {
        addFeatureShortcut("PrevMentFirst=" + fetchFirstWordOrPos(antecedentMent));
      }

      if (!featsToUse.contains("+nolexlastword")) {
        addFeatureShortcut("PrevMentLast=" + fetchLastWordOrPos(antecedentMent));
      }

      if (!featsToUse.contains("+nolexprecedingword")) {
        addFeatureShortcut("PrevMentPreceding=" + fetchPrecedingWordOrPos(antecedentMent));
      }

      if (!featsToUse.contains("+nolexfollowingword")) {
        addFeatureShortcut("PrevMentFollowing=" + fetchFollowingWordOrPos(antecedentMent));
      }


      if (featsToUse.contains("FINAL")) {
        addFeatureShortcut("PrevSynPos=" + antecedentMent.computeSyntacticUnigram);
        addFeatureShortcut("PrevSynPos=" + antecedentMent.computeSyntacticBigram);
      }
      if (featsToUse.contains("MOARPW")) {
        addFeatureShortcut("PrevMentGovernor=" + fetchGovernorWordOrPos(antecedentMent));
      }      
      if (featsToUse.contains("MOARPW")) {
        addFeatureShortcut("PrevSentMentIdx=" + computeSentMentIdx(docGraph, antecedentMent));
      }
      
      if (featsToUse.contains("MOARPW")){
       addFeatureShortcut("AntNerStr=" + antecedentMent.nerString);
       addFeatureShortcut("AntAnimacy=" + AnimacyHelper.getAnimacy(antecedentMent));    
      if (antecedentMent.mentionType == MentionType.PRONOMINAL){
        addFeatureShortcut("AntGend=" + computePronGender(antecedentMent));
      } else {
        addFeatureShortcut("AntGend=" + antecedentMent.gender);
      }
      if (antecedentMent.mentionType == MentionType.PRONOMINAL){
        addFeatureShortcut("AntNumber=" + computePronNumber(antecedentMent));
      } else {
        addFeatureShortcut("AntNumber=" + antecedentMent.number);
      }	        
       addFeatureShortcut("AntPerson=" + computePronPerson(antecedentMent));
       // in BCS they only use number and gender of antecedent and only on closedClass (so we try to mimic their conditions here)
      } else if (featsToUse.contains("FINAL") && mentType.isClosedClass()){ 
        addFeatureShortcut("AntGend=" + antecedentMent.gender);
        addFeatureShortcut("AntNumber=" + antecedentMent.number);
      }

    }
    
    // Features just on currentMent
    if (!startingNew &&  !anaphoricityMode) {

	    if (!featsToUse.contains("+nomentlen")) {
	      addFeatureShortcut("CurrMentLen=" + currMent.words.size);
	    }

	    if (!featsToUse.contains("+nolexanaph")) {
	      addFeatureShortcut("CurrMentHead=" + fetchHeadWordOrPos(currMent));
	    }

	    if (!featsToUse.contains("+nolexfirstword")) {
	      addFeatureShortcut("CurrMentFirst=" + fetchFirstWordOrPos(currMent));
	    }

	    if (!featsToUse.contains("+nolexlastword")) {
	      addFeatureShortcut("CurrMentLast=" + fetchLastWordOrPos(currMent));
	    }

	    if (!featsToUse.contains("+nolexprecedingword")) {
	      addFeatureShortcut("CurrMentPreceding=" + fetchPrecedingWordOrPos(currMent));
	    }

	    if (!featsToUse.contains("+nolexfollowingword")) {
	      addFeatureShortcut("CurrMentFollowing=" + fetchFollowingWordOrPos(currMent));
	    }

	    if (featsToUse.contains("FINAL")) {
	      addFeatureShortcut("CurrSynPos=" + currMent.computeSyntacticUnigram);
	      addFeatureShortcut("CurrSynPos=" + currMent.computeSyntacticBigram);
	    }
	    
        if (featsToUse.contains("MOARPW")) {
          addFeatureShortcut("CurrMentGovernor=" + fetchGovernorWordOrPos(currMent));
        }
        
	    if (featsToUse.contains("MOARPW")) {
	       addFeatureShortcut("CurrSentMentIdx=" + computeSentMentIdx(docGraph, currMent));
	    }        
	    
       if (featsToUse.contains("MOARPW")){
	      addFeatureShortcut("CurrNerStr=" + currMent.nerString);
	      addFeatureShortcut("CurrAnimacy=" + AnimacyHelper.getAnimacy(currMent)); 
	      if (currMent.mentionType == MentionType.PRONOMINAL){
	        addFeatureShortcut("CurrGend=" + computePronGender(currMent));
	      } else {
	        addFeatureShortcut("CurrGend=" + currMent.gender);
	      }
	      if (currMent.mentionType == MentionType.PRONOMINAL){
	        addFeatureShortcut("CurrNumber=" + computePronNumber(currMent));
	      } else {
	        addFeatureShortcut("CurrNumber=" + currMent.number);
	      }	      
	      addFeatureShortcut("CurrPerson=" + computePronPerson(currMent));
      }	    
	    
    }    
    
    // Common to all pairs
    if (!startingNew && !anaphoricityMode) {
      // Distance to antecedent
      if (!featsToUse.contains("+nomentdist")) {
        addFeatureShortcut("Dist=" + Math.min(currMentIdx - antecedentIdx, 10));
      }

      if (!featsToUse.contains("+nosentdist")) {
        addFeatureShortcut("SentDist=" + Math.min(currMent.sentIdx - antecedentMent.sentIdx, 10));
      }

      if (featsToUse.contains("FINAL") || featsToUse.contains("+iwi")) {
        addFeatureShortcut("iWi=" + currMent.iWi(antecedentMent));
      }

      if (featsToUse.contains("FINAL") || featsToUse.contains("+altsyn")) {
        addFeatureShortcut("SynPoses=" + currMent.computeSyntacticUnigram + "-" + antecedentMent.computeSyntacticUnigram);
        addFeatureShortcut("SynPoses=" + currMent.computeSyntacticBigram + "-" + antecedentMent.computeSyntacticBigram);
      }
      
      if (featsToUse.contains("FINAL")){
        addFeatureShortcut("SameSpeaker=" + (currMent.speaker == antecedentMent.speaker));
        addFeatureShortcut("DocType=" + (if (docGraph.corefDoc.rawDoc.isConversation) "CONVERSATION" else "ARTICLE"));
      }
      
      
	  val exactStrMatch = (currMent.spanToString.toLowerCase.equals(antecedentMent.spanToString.toLowerCase));
	  
	  if (!featsToUse.contains("+noexactmatch")) {
	     addFeatureShortcut("ExactStrMatch=" + exactStrMatch);
	  }        
      
	   if (featsToUse.contains("+bilexical")){ 
	       // we never counted head-POS pairs, so anything w/ a POS is automatically excluded (which is fine, since
	       // having a POS implies wasn't frequent enough as a unigram, let alone as a pair) 
	       val currentHead = currMent.headString;
	       val antecedentHead = antecedentMent.headString;
	       if (lexicalCounts.commonBilexPairs.containsKey((currentHead,antecedentHead))){
	         addFeatureShortcut("Heads=" + currentHead + "-" + antecedentHead);
	       } else {
	         addFeatureShortcut("Heads=BACKOFF_INDICATOR");
	       }
	    }              
      
    }
    
    // Closed class (mostly pronoun) specific features
    // We only separate these b/c we compute these things differently for prons
    if (mentType.isClosedClass) {
      if (featsToUse.contains("MOARPW") && !startingNew && antecedentMent.mentionType == MentionType.PRONOMINAL) {
        addFeatureShortcut("Numbs=" + computePronNumber(currMent) + "," + computePronNumber(antecedentMent));
        addFeatureShortcut("Gends=" + computePronGender(currMent) + "," + computePronGender(antecedentMent));
        addFeatureShortcut("Persons=" + computePronPerson(currMent) + "," + computePronPerson(antecedentMent));        
      }
    }
    // Nominal and proper-specific features
    if (!mentType.isClosedClass) {
      if (!startingNew) {
        if (featsToUse.contains("FINAL") || featsToUse.contains("+emcontained")) {
          addFeatureShortcut("ThisContained=" + (antecedentMent.spanToString.contains(currMent.spanToString)));
          addFeatureShortcut("AntContained=" + (currMent.spanToString.contains(antecedentMent.spanToString)));
        }
        // Head match
        val headMatch = currMent.headStringLc.equals(antecedentMent.headString.toLowerCase);
        
        if (!featsToUse.contains("+noheadmatch")) {
          addFeatureShortcut("ExactHeadMatch=" + headMatch);
        }

        if (featsToUse.contains("FINAL") || featsToUse.contains("+hmcontained")) {
          addFeatureShortcut("ThisHeadContained=" + (antecedentMent.spanToString.contains(currMent.headString)));
          addFeatureShortcut("AntHeadContained=" + (currMent.spanToString.contains(antecedentMent.headString)));
        }
        // Agreement
        if (featsToUse.contains("MOARPW")) {
          addFeatureShortcut("Gends=" + currMent.gender + "," + antecedentMent.gender);
          addFeatureShortcut("Numbs=" + currMent.number + "," + antecedentMent.number);
          addFeatureShortcut("Nerts=" + currMent.nerString + "," + antecedentMent.nerString);
          addFeatureShortcut("Persons=" + computePronPerson(currMent) + "," + computePronPerson(antecedentMent)); 
        }
      }
    }
	// again adding raw features that features in BCS are conjoined with (but we don't conjoin with them)
    if (!startingNew && !anaphoricityMode){
        addFeatureShortcut("CurrPronConjStr=" + currMent.computeCanonicalPronounsConjunctionStr);
        addFeatureShortcut("PrevPronConjStr=" + antecedentMent.computeCanonicalPronounsConjunctionStr);
    }
    if (featsToUse.contains("MOARPW")){
      addFeatureShortcut("TypeStrs=" + currMent.computeCanonicalPronounsConjunctionStr + "-" + antecedentMent.computeCanonicalPronounsConjunctionStr);
    }
    feats.toArray
  }
  
  private def fetchHeadWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.headStringLc, ment.pos(ment.headIdx - ment.startIdx), lexicalCounts.commonHeadWordCounts);
  private def fetchFirstWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.words(0).toLowerCase, ment.pos(0), lexicalCounts.commonFirstWordCounts);
  
  private def fetchLastWordOrPos(ment: Mention) = {
    if (ment.words.size == 1 || ment.endIdx - 1 == ment.headIdx) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(ment.words.size - 1).toLowerCase, ment.pos(ment.pos.size - 1), lexicalCounts.commonLastWordCounts);
    }
  }
  private def fetchPenultimateWordOrPos(ment: Mention) = {
    if (ment.words.size <= 2) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(ment.words.size - 2).toLowerCase, ment.pos(ment.pos.size - 2), lexicalCounts.commonPenultimateWordCounts);
    }
  }
  private def fetchSecondWordOrPos(ment: Mention) = {
    if (ment.words.size <= 3) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(1).toLowerCase, ment.pos(1), lexicalCounts.commonSecondWordCounts);
    }
  }
  
  private def fetchPrecedingWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(-1).toLowerCase, ment.contextPosOrPlaceholder(-1), lexicalCounts.commonPrecedingWordCounts);
  private def fetchFollowingWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(ment.words.size).toLowerCase, ment.contextPosOrPlaceholder(ment.words.size), lexicalCounts.commonFollowingWordCounts);
  private def fetchPrecedingBy2WordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(-2).toLowerCase, ment.contextPosOrPlaceholder(-2), lexicalCounts.commonPrecedingBy2WordCounts);
  private def fetchFollowingBy2WordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(ment.words.size + 1).toLowerCase, ment.contextPosOrPlaceholder(ment.words.size + 1), lexicalCounts.commonFollowingBy2WordCounts);
  private def fetchGovernorWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.governor.toLowerCase, ment.governorPos, lexicalCounts.commonGovernorWordCounts);
  
  
  private def fetchWordOrPosDefault(word: String, pos: String, counter: Counter[String]) = {
    if (counter.containsKey(word)) {
      word;
    } else if (featsToUse.contains("+NOPOSBACKOFF")) {
      ""
    } else {
      pos;
    }
  }
    
  private def fetchHeadWord(ment: Mention) = ment.words(ment.headIdx - ment.startIdx);
  private def fetchFirstWord(ment: Mention) = ment.words(0);
  private def fetchLastWord(ment: Mention) = ment.words(ment.pos.size - 1);
  private def fetchPrecedingWord(ment: Mention) = ment.contextWordOrPlaceholder(-1);
  private def fetchFollowingWord(ment: Mention) = ment.contextWordOrPlaceholder(ment.pos.size);
  
  
  private def computePronNumber(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.singularPronouns.contains(ment.headStringLc)) {
      "SING"
    } else if (PronounDictionary.pluralPronouns.contains(ment.headStringLc)) {
      "PLU"
    } else {
      "UNKNOWN"
    }
  }
  
  private def computePronGender(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.malePronouns.contains(ment.headStringLc)) {
      "MALE"
    } else if (PronounDictionary.femalePronouns.contains(ment.headStringLc)) {
      "FEMALE"
    } else if (PronounDictionary.neutralPronouns.contains(ment.headStringLc)) {
      "NEUTRAL"
    } else {
      "UNKNOWN"
    }
  }
  
  private def computePronPerson(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.firstPersonPronouns.contains(ment.headStringLc)) {
      "1st"
    } else if (PronounDictionary.secondPersonPronouns.contains(ment.headStringLc)) {
      "2nd"
    } else if (PronounDictionary.firstPersonPronouns.contains(ment.headStringLc)) {
      "3rd"
    } else {
      "OTHER"
    }
  }
  
  private def computeSentMentIdx(docGraph: DocumentGraph, ment: Mention) = {
    var currIdx = ment.mentIdx - 1;
    while (currIdx >= 0 && docGraph.getMention(currIdx).sentIdx == ment.sentIdx) {
      currIdx -= 1;
    }
    ment.mentIdx - currIdx;
  }

}
  
object SeparatingFeaturizer {
  val UnkFeatName = "UNK_FEAT"; // BCS adds UNK_FEATS; we ignore these in general, but keep them here for compatibility
}
