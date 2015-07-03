//package edu.berkeley.nlp.coref
//import java.net.URL
//import edu.mit.jwi.item.IIndexWord
//import edu.mit.jwi.item.IWord
//import edu.mit.jwi.item.IWordID
//import edu.mit.jwi.Dictionary
//import edu.mit.jwi.item.POS
//import edu.mit.jwi.morph.WordnetStemmer
//import edu.mit.jwi.item.ISynset
//import scala.collection.JavaConverters._
//import edu.mit.jwi.item.Pointer
//import scala.collection.mutable.ArrayBuffer
//import scala.collection.mutable.HashSet
//import edu.mit.jwi.RAMDictionary
//import edu.mit.jwi.data.ILoadPolicy
//import edu.berkeley.nlp.futile.util.Logger
//
//class WordNetInterfacer(path: String) {
//  val url = new URL("file", null, path);
//  
////  val dict = new Dictionary(url);
////  dict.open();
//  val originalDict = new Dictionary(url);
//  originalDict.open();
//  val dict = new RAMDictionary(originalDict, ILoadPolicy.IMMEDIATE_LOAD);
//  dict.open();
//  
//  val wns = new WordnetStemmer(dict);
//  
//  def getLemmas(head: String): Set[String] = {
//    getNounStemSet(head);
//  }
//  
//  def getSynonyms(head: String): Set[String] = {
//    getNounStemSet(head).flatMap((headStem: String) => {
//      val wordSynset = getWordSynset(headStem);
//      if (wordSynset != null) wordSynset.getWords().asScala.map(_.getLemma()) else Set[String]();
//    });
//  }
//  
//  def getHypernyms(head: String): Set[String] = {
//    val initialSynset = getNounStemSet(head).flatMap((headStem: String) => {
//      if (getWordSynset(headStem) != null) Set[ISynset](getWordSynset(headStem)) else Set[ISynset]();
//    }).toSet
//    getHypernyms(10, initialSynset).flatMap(_.getWords().asScala.map(_.getLemma())).toSet;
//  }
//  
//  def areSynonyms(firstHead: String, secondHead: String) = {
//    val stemsFirstHead = getNounStemSet(firstHead);
//    val stemsSecondHead = getNounStemSet(secondHead);
//    var isSynonym = false;
//    for (wordAStem <- stemsFirstHead) {
//      val wordASynset: ISynset = getWordSynset(wordAStem);
//      if (wordASynset != null) {
//        for (wordBStem <- stemsSecondHead) {
//          isSynonym = isSynonym || wordASynset.getWords().asScala.map(_.getLemma()).contains(wordBStem);
//        }
//      }
//    }
//    isSynonym
//  }
//  
//  def areHypernyms(head: String, possibleHypernym: String) = {
//    val stemsHead = getNounStemSet(head);
//    val stemsPossibleHypernym = getNounStemSet(possibleHypernym);
//    var isHypernym = false;
//    for (headStem <- stemsHead) {
//      val headSynset: ISynset = getWordSynset(headStem);
//      if (headSynset != null) {
//        // 10 levels in the tree should be enough for anybody...
//        val hypernyms = getHypernyms(10, Set(headSynset));
//        for(hypernym <- hypernyms){
//          val hypernymWords = hypernym.getWords();
//          for (i <- 0 until hypernymWords.size()) {
//            isHypernym = isHypernym || stemsPossibleHypernym.contains(hypernymWords.get(i).getLemma());
//          }
//        }
//      }
//    }
//    isHypernym
//  }
//  
//  private def getHypernyms(numLevelsToRecurse: Int, synsets: Set[ISynset]): HashSet[ISynset] = {
//    var synsetsThisLevel = new HashSet[ISynset]() ++ synsets;
//    var synsetsNextLevel = new HashSet[ISynset]();
//    val allSynsets = new HashSet[ISynset]();
//    for (i <- 0 until numLevelsToRecurse) {
//      if (!synsetsThisLevel.isEmpty) {
//        for (synset <- synsetsThisLevel) {
//          synsetsNextLevel ++= synset.getRelatedSynsets(Pointer.HYPERNYM).asScala.map(dict.getSynset(_));
//        }
//        // Don't visit nodes we've already been to
//        synsetsThisLevel = (synsetsNextLevel -- allSynsets);
//        allSynsets ++= synsetsNextLevel;
//        synsetsNextLevel = new HashSet[ISynset]();
//      }
//    }
//    allSynsets;
//  }
//  
//  private def getWordSynset(stemmedWord: String) = {
//    val idxWord: IIndexWord = dict.getIndexWord(stemmedWord, POS.NOUN);
//    if (idxWord != null) {
//      val wordID: IWordID = idxWord.getWordIDs().get(0);
//      val word: IWord = dict.getWord(wordID);
//      word.getSynset();
//    } else {
//      null;
//    }
//  }
//  
//  private def getNounStemSet(head: String): Set[String] = {
//    require(head != null && !head.isEmpty());
//    var toReturn = Set[String]();
//    try {
//      toReturn = wns.findStems(head, POS.NOUN).asScala.toSet;
//    } catch {
//      case e: IllegalArgumentException => Logger.logss("IllegalArgumentException on " + head);
//      case _ => Logger.logss("Badness"); System.exit(0);
//    }
//    toReturn;
//  }
//
//}
//
//object WordNetInterfacer {
//  
//  
//  
//  
//  def main(args: Array[String]) = {
//    val path = "/Users/gdurrett/Documents/Berkeley/Utils/WNdb-3.0/dict/";
//    val url = new URL("file", null, path);
//    
//    val dict = new Dictionary(url);
//    dict.open();
//    val idxWord: IIndexWord = dict.getIndexWord("dog", POS.NOUN);
//    val wordID: IWordID = idxWord.getWordIDs().get(0);
//    val word: IWord = dict.getWord(wordID);
//    println("Id = " + wordID);
//    println("Lemma = " + word.getLemma());
//    println("Gloss = " + word.getSynset().getGloss());
//    
//    val synset: ISynset = word.getSynset();
//    // iterate over words associated with the synset
//    println("Synonyms");
//    synset.getWords().asScala.foreach(word => println(word.getLemma()))
//    
//    val hypernyms = synset.getRelatedSynsets(Pointer.HYPERNYM);
//    println("Hypernyms");
//    for(sid <- hypernyms.asScala){
//      println(sid + ": " + dict.getSynset(sid).getWords().asScala.map(_.getLemma()));
//    }
//    
//    val wns = new WordnetStemmer(dict);
//    println(wns.findStems("dogs", POS.NOUN));
//    println(wns.findStems("DOGS", POS.NOUN));
//    println(wns.findStems("Presidents", POS.NOUN));
//    
//    
//    println("===============");
//    val wordNetInterfacer = new WordNetInterfacer(path);
//    println("Synonyms: dog cat? (should be false) " + wordNetInterfacer.areSynonyms("dog", "cat"));
//    println("Synonyms: dog domestic_dog? (should be true) " + wordNetInterfacer.areSynonyms("dog", "domestic_dog"));
//    
//    
//    println("Hypernyms: dog domestic_dog? (should be false) " + wordNetInterfacer.areHypernyms("dog", "domestic_dog"));
//    println("Hypernyms: dog canine? (should be true) " + wordNetInterfacer.areHypernyms("dog", "canine"));
//    println("Hypernyms: canine dog? (should be false) " + wordNetInterfacer.areHypernyms("canine", "dog"));
//    
//    
//    println("===============");
//    println(wordNetInterfacer.getLemmas("dog"));
//    println(wordNetInterfacer.getSynonyms("dog"));
//    println(wordNetInterfacer.getSynonyms("cat"));
//    println(wordNetInterfacer.getHypernyms("cat"));
//    
//  }
//}