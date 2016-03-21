package edu.berkeley.nlp.coref
import edu.berkeley.nlp.futile.syntax.Tree

// Chunks are semi-inclusive intervals.
case class Chunk[T](val start: Int,
                    val end: Int,
                    val label: T);

// rawText should only be used to save trouble when outputting the document
// for scoring; never at any other time!
case class ConllDoc(val docID: String,
                    val docPartNo: Int,
                    val words: Seq[Seq[String]],
                    val pos: Seq[Seq[String]],
                    val trees: Seq[DepConstTree],
                    val nerChunks: Seq[Seq[Chunk[String]]],
                    val corefChunks: Seq[Seq[Chunk[Int]]],
                    val speakers: Seq[Seq[String]],
                    val rawText: Seq[Seq[String]]) {
  
  val numSents = words.size;
  
  // updating...blah
  val allSpeakers = scala.collection.mutable.Set[String]();
  var gatheredSpeakers = false;
  
  def getSpeakers():scala.collection.mutable.Set[String] = {
    if (gatheredSpeakers){
      return allSpeakers;
    } else {
      for (speakerSent <- speakers){
        for (speaker <- speakerSent){
          allSpeakers.add(speaker.replace("-","").replace("_","").replace(".","").toLowerCase);
        }
      }
      gatheredSpeakers = true;
      return allSpeakers;
    }
  }
  
  def printableDocName = docID + " (part " + docPartNo + ")";
  
  def isConversation = docID.startsWith("bc") || docID.startsWith("wb");
}