package edu.harvard.nlp.moarcoref

import java.io.File
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.ConllDoc
import edu.berkeley.nlp.coref.ConllDocReader
import edu.berkeley.nlp.coref.CorefDoc
import edu.berkeley.nlp.coref.CorefDocAssembler
import edu.berkeley.nlp.coref.NumberGenderComputer
import edu.berkeley.nlp.coref.MentionPropertyComputer

object SmallerCorefSystem {
  
  def loadRawConllDocs(path: String, size: Int, gold: Boolean): Seq[ConllDoc] = {
    val suffix = if (gold) "gold_conll" else MiniDriver.docSuffix;
    Logger.logss("Loading " + size + " docs from " + path + " ending with " + suffix);
    val files = new File(path).listFiles().filter(file => file.getAbsolutePath.endsWith(suffix));
    val reader = new ConllDocReader(MiniDriver.lang);
    val docs = new ArrayBuffer[ConllDoc];
    var docCounter = 0;
    var fileIdx = 0;
    while (fileIdx < files.size && (size == -1 || docCounter < size)) {
      val newDocs = reader.readConllDocs(files(fileIdx).getAbsolutePath);
      docs ++= newDocs;
      docCounter += newDocs.size
      fileIdx += 1;
    }
    val numDocs = if (size == -1) docs.size else Math.min(size, files.size);
    Logger.logss(docs.size + " docs loaded from " + fileIdx + " files, retaining " + numDocs);
    if (docs.size == 0) {
      Logger.logss("WARNING: Zero docs loaded...double check your paths unless you meant for this happen");
    }
    val docsToUse = docs.slice(0, numDocs);
    
    docsToUse;
  }
  
  def loadCorefDocs(path: String, size: Int, numberGenderComputer: NumberGenderComputer, gold: Boolean): Seq[CorefDoc] = {
    val docs = loadRawConllDocs(path, size, gold);
    val assembler = CorefDocAssembler(MiniDriver.lang, MiniDriver.useGoldMentions);
    val mentionPropertyComputer = new MentionPropertyComputer(numberGenderComputer);
    val corefDocs = docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    CorefDoc.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
}