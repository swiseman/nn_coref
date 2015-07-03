package edu.berkeley.nlp.coref
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.HashMap
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.coref.sem.QueryCountsBundle
import edu.berkeley.nlp.coref.preprocess.NerExample

@SerialVersionUID(1L)
class PairwiseIndexingFeaturizerJoint(val featureIndexer: Indexer[String],
                                      val featsToUse: String,
                                      val conjType: ConjType,
                                      val lexicalCounts: LexicalCountsBundle,
                                      val queryCounts: QueryCountsBundle) extends PairwiseIndexingFeaturizer with Serializable {
  def getIndexer = featureIndexer;
  
  def getIndex(feature: String, addToFeaturizer: Boolean): Int = {
    if (!addToFeaturizer) {
      if (!featureIndexer.contains(feature)) {
        val idx = featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
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
    def addFeatureShortcut = (featName: String) => {
      // Only used in CANONICAL_ONLY_PAIR, so only compute the truth value in this case
      val isPairFeature = conjType == ConjType.CANONICAL_ONLY_PAIR_CONJ && !(featName.startsWith("SN") || featName.startsWith("PrevMent"));
      addFeatureAndConjunctions(feats, featName, currMent, antecedentMent, isPairFeature, addToFeaturizer);
    }
    // Features on anaphoricity
    val mentType = currMent.mentionType;
    val startingNew = antecedentIdx == currMentIdx;
    // When using very minimal feature sets, you might need to include this so every decision
    // has at least one feature over it.
    if (featsToUse.contains("+bias")) {
      addFeatureShortcut("SN=" + startingNew);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nomentlen")) {
//      addFeatureShortcut("SNMentLen=" + currMent.spanToString.split("\\s+").size + "-SN=" + startingNew);
      addFeatureShortcut("SNMentLen=" + currMent.words.size + "-SN=" + startingNew);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nolexanaph") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentHead=" + fetchHeadWordOrPos(currMent) + "-SN=" + startingNew);
      if (featsToUse.contains("+wordbackoff")) {
        val word = fetchHeadWord(antecedentMent);
        val featStart = "SNMentHead";
//        addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Shape=" + fetchShape(word) + "-SN=" + startingNew);
        addFeatureShortcut(featStart + "Class=" + fetchClass(word) + "-SN=" + startingNew);
      }
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nolexfirstword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentFirst=" + fetchFirstWordOrPos(currMent) + "-SN=" + startingNew);
      if (featsToUse.contains("+wordbackoff")) {
        val word = fetchFirstWord(antecedentMent);
        val featStart = "SNMentFirst";
//        addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Shape=" + fetchShape(word) + "-SN=" + startingNew);
        addFeatureShortcut(featStart + "Class=" + fetchClass(word) + "-SN=" + startingNew);
      }
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nolexlastword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentLast=" + fetchLastWordOrPos(currMent) + "-SN=" + startingNew);
      if (featsToUse.contains("+wordbackoff")) {
        val word = fetchLastWord(antecedentMent);
        val featStart = "SNMentLast";
//        addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Shape=" + fetchShape(word) + "-SN=" + startingNew);
        addFeatureShortcut(featStart + "Class=" + fetchClass(word) + "-SN=" + startingNew);
      }
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nolexprecedingword")) {
      addFeatureShortcut("SNMentPreceding=" + fetchPrecedingWordOrPos(currMent) + "-SN=" + startingNew);
      if (featsToUse.contains("+wordbackoff")) {
        val word = fetchPrecedingWord(antecedentMent);
        val featStart = "SNMentPreceding";
//        addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Shape=" + fetchShape(word) + "-SN=" + startingNew);
        addFeatureShortcut(featStart + "Class=" + fetchClass(word) + "-SN=" + startingNew);
      }
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("+nolexfollowingword")) {
      addFeatureShortcut("SNMentFollowing=" + fetchFollowingWordOrPos(currMent) + "-SN=" + startingNew);
      if (featsToUse.contains("+wordbackoff")) {
        val word = fetchFollowingWord(antecedentMent);
        val featStart = "SNMentFollowing";
//        addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word) + "-SN=" + startingNew);
//        addFeatureShortcut(featStart + "Shape=" + fetchShape(word) + "-SN=" + startingNew);
        addFeatureShortcut(featStart + "Class=" + fetchClass(word) + "-SN=" + startingNew);
      }
    }
    if (featsToUse.contains("+lexpenultimateword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentPen=" + fetchPenultimateWordOrPos(currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+lexsecondword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentSecond=" + fetchSecondWordOrPos(currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+lexprecedingby2word")) {
      addFeatureShortcut("SNMentPrecedingBy2=" + fetchPrecedingBy2WordOrPos(currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+lexfollowingby2word")) {
      addFeatureShortcut("SNMentFollowingBy2=" + fetchFollowingBy2WordOrPos(currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+lexgovernor")) {
      addFeatureShortcut("SNGovernor=" + fetchGovernorWordOrPos(currMent) + "-SN=" + startingNew);
    }
    // N.B. INCLUDED IN FINAL
    if (featsToUse.contains("FINAL") || featsToUse.contains("+altsyn")) {
      addFeatureShortcut("SNSynPos=" + currMent.computeSyntacticUnigram);
      addFeatureShortcut("SNSynPos=" + currMent.computeSyntacticBigram);
    }
    if (featsToUse.contains("+sentmentidx")) {
      addFeatureShortcut("SNSentMentIdx=" + computeSentMentIdx(docGraph, currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+latent")) {
      for (clustererIdx <- 0 until docGraph.numClusterers) {
        addFeatureShortcut("SNTopicC" + clustererIdx + "=" + computeTopicLabel(docGraph, clustererIdx, currMentIdx) + "-SN=" + startingNew);
      }
    }
    if (featsToUse.contains("+def") && antecedentMent.mentionType != MentionType.PRONOMINAL) {
      addFeatureShortcut("SNDef=" + computeDefiniteness(currMent) + "-SN=" + startingNew);
    }
    if (featsToUse.contains("+synpos")) {
      addFeatureShortcut("SNSynPos=" + currMent.computeSyntacticPosition + "-SN=" + startingNew);
    }
    
    
    // Features just on the antecedent
    if (!startingNew) {
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nomentlen")) {
//        addFeatureShortcut("PrevMentLen=" + antecedentMent.spanToString.split("\\s+").size);
        addFeatureShortcut("PrevMentLen=" + antecedentMent.words.size);
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nolexanaph") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevMentHead=" + fetchHeadWordOrPos(antecedentMent));
        if (featsToUse.contains("+wordbackoff")) {
          val word = fetchHeadWord(antecedentMent);
          val featStart = "PrevMentHead";
//          addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word));
//          addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word));
//          addFeatureShortcut(featStart + "Shape=" + fetchShape(word));
          addFeatureShortcut(featStart + "Class=" + fetchClass(word));
        }
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nolexfirstword") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevMentFirst=" + fetchFirstWordOrPos(antecedentMent));
        if (featsToUse.contains("+wordbackoff")) {
          val word = fetchFirstWord(antecedentMent);
          val featStart = "PrevMentFirst";
//          addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word));
//          addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word));
//          addFeatureShortcut(featStart + "Shape=" + fetchShape(word));
          addFeatureShortcut(featStart + "Class=" + fetchClass(word));
        }
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nolexlastword") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevMentLast=" + fetchLastWordOrPos(antecedentMent));
        if (featsToUse.contains("+wordbackoff")) {
          val word = fetchLastWord(antecedentMent);
          val featStart = "PrevMentLast";
//          addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word));
//          addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word));
//          addFeatureShortcut(featStart + "Shape=" + fetchShape(word));
          addFeatureShortcut(featStart + "Class=" + fetchClass(word));
        }
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nolexprecedingword")) {
        addFeatureShortcut("PrevMentPreceding=" + fetchPrecedingWordOrPos(antecedentMent));
        if (featsToUse.contains("+wordbackoff")) {
          val word = fetchPrecedingWord(antecedentMent);
          val featStart = "PrevMentPreceding";
//          addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word));
//          addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word));
//          addFeatureShortcut(featStart + "Shape=" + fetchShape(word));
          addFeatureShortcut(featStart + "Class=" + fetchClass(word));
        }
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nolexfollowingword")) {
        addFeatureShortcut("PrevMentFollowing=" + fetchFollowingWordOrPos(antecedentMent));
        if (featsToUse.contains("+wordbackoff")) {
          val word = fetchFollowingWord(antecedentMent);
          val featStart = "PrevMentFollowing";
//          addFeatureShortcut(featStart + "Prefix=" + fetchPrefix(word));
//          addFeatureShortcut(featStart + "Suffix=" + fetchSuffix(word));
//          addFeatureShortcut(featStart + "Shape=" + fetchShape(word));
          addFeatureShortcut(featStart + "Class=" + fetchClass(word));
        }
      }
      if (featsToUse.contains("+lexpenultimateword") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevMentPen=" + fetchPenultimateWordOrPos(antecedentMent));
      }
      if (featsToUse.contains("+lexsecondword") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevMentSecond=" + fetchSecondWordOrPos(antecedentMent));
      }
      if (featsToUse.contains("+lexprecedingby2word")) {
        addFeatureShortcut("PrevMentPrecedingBy2=" + fetchPrecedingBy2WordOrPos(antecedentMent));
      }
      if (featsToUse.contains("+lexfollowingby2word")) {
        addFeatureShortcut("PrevMentFollowingBy2=" + fetchFollowingBy2WordOrPos(antecedentMent));
      }
      if (featsToUse.contains("+lexgovernor")) {
        addFeatureShortcut("PrevMentGovernor=" + fetchGovernorWordOrPos(antecedentMent));
      }
      // N.B. INCLUDED IN FINAL
      if (featsToUse.contains("FINAL") || featsToUse.contains("+altsyn")) {
        addFeatureShortcut("PrevSynPos=" + antecedentMent.computeSyntacticUnigram);
        addFeatureShortcut("PrevSynPos=" + antecedentMent.computeSyntacticBigram);
      }
      if (featsToUse.contains("+sentmentidx")) {
        addFeatureShortcut("PrevSentMentIdx=" + computeSentMentIdx(docGraph, antecedentMent));
      }
      if (featsToUse.contains("+latent")) {
        for (clustererIdx <- 0 until docGraph.numClusterers) {
          addFeatureShortcut("PrevTopicC" + clustererIdx + "=" + computeTopicLabel(docGraph, clustererIdx, antecedentIdx));
        }
      }
      if (featsToUse.contains("+def") && !antecedentMent.mentionType.isClosedClass) {
        addFeatureShortcut("PrevDef=" + computeDefiniteness(antecedentMent));
      }
      if (featsToUse.contains("+synpos")) {
        addFeatureShortcut("PrevSynPos=" + antecedentMent.computeSyntacticPosition);
      }
    }
    
    // Common to all pairs
    if (!startingNew) {
      // Distance to antecedent
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nomentdist")) {
        addFeatureShortcut("Dist=" + Math.min(currMentIdx - antecedentIdx, 10));
      }
      if (featsToUse.contains("+mentdistfine")) {
        val mentDist = currMentIdx - antecedentIdx;
        val bucketedMentDist = if (mentDist >= 10) Math.min(mentDist/10, 5) * 10 else mentDist;
        addFeatureShortcut("Dist=" + mentDist);
      }
      // N.B. INCLUDED IN SURFACE
      if (!featsToUse.contains("+nosentdist")) {
        addFeatureShortcut("SentDist=" + Math.min(currMent.sentIdx - antecedentMent.sentIdx, 10));
      }
      // N.B. INCLUDED IN FINAL
      if (featsToUse.contains("FINAL") || featsToUse.contains("+iwi")) {
        addFeatureShortcut("iWi=" + currMent.iWi(antecedentMent));
      }
      // N.B. INCLUDED IN FINAL
      if (featsToUse.contains("FINAL") || featsToUse.contains("+altsyn")) {
        addFeatureShortcut("SynPoses=" + currMent.computeSyntacticUnigram + "-" + antecedentMent.computeSyntacticUnigram);
        addFeatureShortcut("SynPoses=" + currMent.computeSyntacticBigram + "-" + antecedentMent.computeSyntacticBigram);
      }
      if (featsToUse.contains("+latent")) {
        for (clustererIdx <- 0 until docGraph.numClusterers) {
          val thisLabel = computeTopicLabel(docGraph, clustererIdx, currMentIdx);
          val antLabel = computeTopicLabel(docGraph, clustererIdx, antecedentIdx);
          addFeatureShortcut("Topic=C" + clustererIdx + "-" + thisLabel);
          addFeatureShortcut("Topics=C" + clustererIdx + "-" + thisLabel + "-" + antLabel);
        }
      }
//      if (featsToUse.contains("+synpos")) {
//        addFeatureShortcut("SynPoses=" + currMent.computeSyntacticPosition + "-" + antecedentMent.computeSyntacticPosition);
//      }
//      if (featsToUse.contains("+lexcontexts")) {
//        addFeatureShortcut("CurrPrecPrevPrec=" + fetchPrecedingWordOrPos(currMent) + "-" + fetchPrecedingWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrFollPrevPrec=" + fetchFollowingWordOrPos(currMent) + "-" + fetchPrecedingWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrPrecPrevFoll=" + fetchPrecedingWordOrPos(currMent) + "-" + fetchFollowingWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrFollPrevFoll=" + fetchFollowingWordOrPos(currMent) + "-" + fetchFollowingWordOrPos(antecedentMent));
//      }
    }
    // Closed class (mostly pronoun) specific features
    if (mentType.isClosedClass) {
      // Pronominal features
      // N.B. INCLUDED IN FINAL
      if ((featsToUse.contains("FINAL") || featsToUse.contains("+prongendnum")) && !startingNew) {
        addFeatureShortcut("AntGend=" + antecedentMent.gender);
        addFeatureShortcut("AntNumb=" + antecedentMent.number);
      }
//      if (featsToUse.contains("+customprongendnum") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("AntGend=" + antecedentMent.gender + "-" + currMent.headStringLc);
//        addFeatureShortcut("AntNumb=" + antecedentMent.number + "-" + currMent.headStringLc);
//      }
      // N.B. INCLUDED IN FINAL
      if ((featsToUse.contains("FINAL") || featsToUse.contains("+speaker")) && !startingNew) {
        if (antecedentMent.mentionType == MentionType.PRONOMINAL) {
          addFeatureShortcut("SameSpeaker=" + (if (docGraph.corefDoc.rawDoc.isConversation) "CONVERSATION" else "ARTICLE") +
                             "-" + (currMent.speaker == antecedentMent.speaker));
        }
      }
//      if (featsToUse.contains("+manpronpron") && !startingNew && antecedentMent.mentionType == MentionType.PRONOMINAL) {
//        addFeatureShortcut("PronNumbers=" + computePronNumber(currMent) + "-" + computePronNumber(antecedentMent));
//        addFeatureShortcut("PronGenders=" + computePronGender(currMent) + "-" + computePronGender(antecedentMent));
//        addFeatureShortcut("PronPersons=" + computePronPerson(currMent) + "-" + computePronPerson(antecedentMent));
//      }
//      if (featsToUse.contains("+manpgn") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("PronGendAntGend=" + computePronGender(currMent) + "-" + antecedentMent.gender);
//        addFeatureShortcut("PronNumbAntNumb=" + computePronNumber(currMent) + "-" + antecedentMent.number);
//      }
//      if (featsToUse.contains("+pronpronpair") && !startingNew && antecedentMent.mentionType == MentionType.PRONOMINAL) {
//        addFeatureShortcut("PronPron=" + currMent.headStringLc + "-" + antecedentMent.headStringLc);
//      }
//      if (featsToUse.contains("+proncanonproncanonpair") && !startingNew && antecedentMent.mentionType == MentionType.PRONOMINAL) {
//        addFeatureShortcut("PronCPronC=" + currMent.computeCanonicalPronounsConjunctionStr + "-" + antecedentMent.computeCanonicalPronounsConjunctionStr);
//      }
//      if (featsToUse.contains("+pronprevcontext") && !startingNew) {
//        addFeatureShortcut("CurrPronPrevPreceding=" + currMent.headStringLc + "-" + fetchPrecedingWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrPronPrevFollowing=" + currMent.headStringLc + "-" + fetchFollowingWordOrPos(antecedentMent));
//      }
//      if (featsToUse.contains("+pronnomheadpair") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("CurrPronPrevHead=" + currMent.headStringLc + "-" + fetchHeadWordOrPos(antecedentMent));
//      }
//      if (featsToUse.contains("+pronnomfirstword") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("CurrPronPrevFirst=" + currMent.headStringLc + "-" + fetchFirstWordOrPos(antecedentMent));
//      }
//      if (featsToUse.contains("+proncanonnomheadpair") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("CurrPronCPrevHead=" + currMent.computeCanonicalPronounsConjunctionStr + "-" + fetchHeadWordOrPos(antecedentMent));
//      }
//      if (featsToUse.contains("+proncanonnomfirstword") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("CurrPronCPrevFirst=" + currMent.computeCanonicalPronounsConjunctionStr + "-" + fetchFirstWordOrPos(antecedentMent));
//      }
//      if (featsToUse.contains("+pronnomall") && !startingNew && antecedentMent.mentionType != MentionType.PRONOMINAL) {
//        addFeatureShortcut("CurrPronPrevHead=" + currMent.headStringLc + "-" + fetchHeadWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrPronPrevFirst=" + currMent.headStringLc + "-" + fetchFirstWordOrPos(antecedentMent));
//        addFeatureShortcut("CurrPronPrevLen=" + currMent.headStringLc + "-" + antecedentMent.words.size);
//      }
    }
    // Nominal and proper-specific features
    if (!mentType.isClosedClass) {
      if (!startingNew) {
        // Nominal and proper features
        // String match
        val exactStrMatch = (currMent.spanToString.toLowerCase.equals(antecedentMent.spanToString.toLowerCase));
        // N.B. INCLUDED IN SURFACE
        if (!featsToUse.contains("+noexactmatch")) {
          addFeatureShortcut("ExactStrMatch=" + exactStrMatch);
        }
        // N.B. INCLUDED IN FINAL
        if (featsToUse.contains("FINAL") || featsToUse.contains("+emcontained")) {
          addFeatureShortcut("ThisContained=" + (antecedentMent.spanToString.contains(currMent.spanToString)));
          addFeatureShortcut("AntContained=" + (currMent.spanToString.contains(antecedentMent.spanToString)));
        }
        // Head match
        val headMatch = currMent.headStringLc.equals(antecedentMent.headString.toLowerCase);
        // N.B. INCLUDED IN SURFACE
        if (!featsToUse.contains("+noheadmatch")) {
          addFeatureShortcut("ExactHeadMatch=" + headMatch);
        }
        if (featsToUse.contains("+lexhm")) {
          addFeatureShortcut("LexHeadMatchCurr=" + headMatch + "-" + fetchHeadWordOrPos(currMent));
          addFeatureShortcut("LexHeadMatchPrev=" + headMatch + "-" + fetchHeadWordOrPos(antecedentMent));
        }
        // N.B. INCLUDED IN FINAL
        if (featsToUse.contains("FINAL") || featsToUse.contains("+hmcontained")) {
          addFeatureShortcut("ThisHeadContained=" + (antecedentMent.spanToString.contains(currMent.headString)));
          addFeatureShortcut("AntHeadContained=" + (currMent.spanToString.contains(antecedentMent.headString)));
        }
        // Agreement
        if (featsToUse.contains("+nomgendnum")) {
          addFeatureShortcut("Gends=" + currMent.gender + "," + antecedentMent.gender);
          addFeatureShortcut("Numbs=" + currMent.number + "," + antecedentMent.number);
          addFeatureShortcut("Nerts=" + currMent.nerString + "," + antecedentMent.nerString);
        }
        if (featsToUse.contains("+bilexical")) {
          if (!antecedentMent.mentionType.isClosedClass) {
            addFeatureShortcut("Heads=" + fetchHeadWordOrPos(currMent) + "-" + fetchHeadWordOrPos(antecedentMent));
          }
        }
//        if (featsToUse.contains("+bigovernors")) {
//          if (antecedentMent.mentionType != MentionType.PRONOMINAL) {
//            addFeatureShortcut("CurrHeadPrevGov=" + fetchHeadWordOrPos(currMent) + "-" + fetchGovernorWordOrPos(antecedentMent));
//            addFeatureShortcut("CurrGovPrevHead=" + fetchGovernorWordOrPos(currMent) + "-" + fetchHeadWordOrPos(antecedentMent));
//            addFeatureShortcut("CurrGovPrevGov=" + fetchGovernorWordOrPos(currMent) + "-" + fetchGovernorWordOrPos(antecedentMent));
//          }
//        }
        if (featsToUse.contains("+hearst")) {
          // Only fire on referring linkages that don't have head match
          if (antecedentMent.mentionType != MentionType.PRONOMINAL && currMent.headStringLc != antecedentMent.headStringLc) {
            // Non lower-cased
            val currHead = currMent.headString;
            val antHead = antecedentMent.headString;
            // Order of this call shouldn't matter because we symmetrized the counts
            val pairCount = queryCounts.pairCounts.getCount(antHead -> currHead);
            val present = pairCount > 0.5;
            addFeatureShortcut("HearstPresent=" + present);
            if (present) {
              val logBinnedCountUnnorm = (Math.log(pairCount)/Math.log(10) + 0.5).toInt;
              val logBinnedCountUnnormFine = (Math.log(pairCount)/Math.log(10) * 4 + 0.5).toInt;
              addFeatureShortcut("HearstUnnormBin=" + logBinnedCountUnnorm);
              addFeatureShortcut("HearstUnnormFineBin=" + logBinnedCountUnnormFine);
              var currHeadCount = queryCounts.wordCounts.getCount(currHead);
              var antHeadCount = queryCounts.wordCounts.getCount(antHead);
              // Watch out for divide-by-zero when doing the normalized counts...
              if (currHeadCount == 0 || antHeadCount == 0) {
                Logger.logss("WARNING: Inexplicably, count for " + currHead + " or " + antHead + " is less than the pair count: " +
                             currHeadCount + " " + antHeadCount + " " + pairCount);
                currHeadCount = Math.max(currHeadCount, pairCount);
                antHeadCount = Math.max(antHeadCount, pairCount);
              }
              val logBinnedCountNorm = (Math.log(pairCount/(currHeadCount * antHeadCount))/Math.log(10) + 0.5).toInt;
              val logBinnedCountNormFine = (Math.log(pairCount/(currHeadCount * antHeadCount))/Math.log(10) * 4 + 0.5).toInt;
              addFeatureShortcut("HearstNormBin=" + logBinnedCountNorm);
              addFeatureShortcut("HearstNormFineBin=" + logBinnedCountNormFine);
//              Logger.logss(currHead + " " + antHead + ": " + currHeadCount + " " + antHeadCount + " " + pairCount + " " + logBinnedCountUnnorm + " " + logBinnedCountNorm)
            }
          }
        }
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // ADD YOUR OWN FEATURES HERE!                                                                //
    //   See above for examples of how to do this. Typically use addFeatureShortcut since this    //
    // gives you your feature as well as conjunctions, but you can also directly call             //
    // feats += getIndex(feat, addToFeaturizer);                                                  //
    //                                                                                            //
    // To control feature sets, featsToUse is passed down from pairwiseFeats (the command line    //
    // argument). We currently use magic words all starting with +, but you do have to make       //
    // sure that you don't make a magic word that's a prefix of another, or else both will be     //
    // added when the longer one is.                                                              //
    //                                                                                            //
    // Happy feature engineering!                                                                 //
    ////////////////////////////////////////////////////////////////////////////////////////////////
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
  
  private def fetchPrefix(word: String) = {
    if (word.size >= 3 && lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 3))) {
      word.substring(0, 3);
    } else if (word.size >= 2 && lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 2))) {
      word.substring(0, 2);
    } else if (lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 1))) {
      word.substring(0, 1);
    } else {
      "";
    }
  }
  
  private def fetchSuffix(word: String) = {
    if (word.size >= 3 && lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 3))) {
      word.substring(word.size - 3);
    } else if (word.size >= 2 && lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 2))) {
      word.substring(word.size - 2);
    } else if (lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 1))) {
      word.substring(word.size - 1);
    } else {
      "";
    }
  }
  
  private def fetchShape(word: String) = {
    if (lexicalCounts.commonShapeCounts.containsKey(NerExample.shapeFor(word))) {
      NerExample.shapeFor(word);
    } else {
      "";
    }
  }
  
  private def fetchClass(word: String) = {
    if (lexicalCounts.commonClassCounts.containsKey(NerExample.classFor(word))) {
      NerExample.classFor(word);
    } else {
      "";
    }
  }
  
  private def fetchHeadWord(ment: Mention) = ment.words(ment.headIdx - ment.startIdx);
  private def fetchFirstWord(ment: Mention) = ment.words(0);
  private def fetchLastWord(ment: Mention) = ment.words(ment.pos.size - 1);
  private def fetchPrecedingWord(ment: Mention) = ment.contextWordOrPlaceholder(-1);
  private def fetchFollowingWord(ment: Mention) = ment.contextWordOrPlaceholder(ment.pos.size);
//  
//  private def fetchHeadPos(ment: Mention) = ment.pos(ment.headIdx - ment.startIdx);
//  private def fetchFirstPos(ment: Mention) = ment.pos(0);
//  private def fetchLastPos(ment: Mention) = ment.pos(ment.pos.size - 1);
//  private def fetchPrecedingPos(ment: Mention) = ment.contextPosOrPlaceholder(-1);
//  private def fetchFollowingPos(ment: Mention) = ment.contextPosOrPlaceholder(ment.pos.size);
  
  private def computeDefiniteness(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (firstWord.equals("the")) {
      "DEF"
    } else if (firstWord.equals("a") || firstWord.equals("an")) {
      "INDEF"
    } else {
      "NONE"
    }
  }
  
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
    ment.mentIdx - currIdx + 1;
  }

  def computeTopicLabel(docGraph: DocumentGraph, clustererIdx: Int, mentIdx: Int): String = {
    val ment = docGraph.getMention(mentIdx);
    if (ment.mentionType == MentionType.PRONOMINAL && featsToUse.contains("noprons")) {
      "PRON"
    } else if ((ment.mentionType == MentionType.NOMINAL || ment.mentionType == MentionType.PROPER) && featsToUse.contains("nonomsprops")) {
      "NOMPROP"
    } else {
      docGraph.getBestCluster(clustererIdx, mentIdx) + ""
    }
  }
  
  def computeDistribLabel(docGraph: DocumentGraph, clustererIdx: Int, mentIdx: Int, valIdx: Int): Int = {
    docGraph.storedDistributedLabels(clustererIdx)(mentIdx)(valIdx);
  }
  
  def numDistribLabels(docGraph: DocumentGraph, clustererIdx: Int): Int = {
    docGraph.numClusters(clustererIdx);
  }
}
  
object PairwiseIndexingFeaturizerJoint {
  val UnkFeatName = "UNK_FEAT";
}
