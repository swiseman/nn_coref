package edu.berkeley.nlp.coref.preprocess
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import edu.berkeley.nlp.coref.ConllDoc
import edu.berkeley.nlp.futile.classify.GeneralLogisticRegression
import edu.berkeley.nlp.coref.CorefSystem
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.GUtil
import edu.berkeley.nlp.futile.classify.SequenceExample
import edu.berkeley.nlp.futile.fig.basic.IOUtils


class NerSystem(val featureIndexer: Indexer[String], val weights: Array[Double]) {
  
  val featurizedTransitionMatrix = NerSystem.TransitionMatrix.map(_.map(arr => if (arr != null) arr.map(featureIndexer.getIndex(_)) else null));
  
  def runNerSystem(sentenceWords: Array[String], sentencePos: Array[String]): Array[String] = {
    val example = new NerExample(sentenceWords, sentencePos, null, NerSystem.LabelIndexer);
    val seqExample = new SequenceExample(featurizedTransitionMatrix, example.featurize(featureIndexer, false), null);
    seqExample.decode(weights).map(NerSystem.LabelIndexer.getObject(_));
  }
}

object NerSystem {
  val LabelIndexer = new Indexer[String]();  
  LabelIndexer.getIndex("O");
  LabelIndexer.getIndex("B");
  LabelIndexer.getIndex("I");
  // Disallow O-I
  val TransitionMatrix = Array(Array(Array("Trans=O-O"), Array("Trans=O-B"), null),
                               Array(Array("Trans=B-O"), Array("Trans=B-B"), Array("Trans=B-I")),
                               Array(Array("Trans=I-O"), Array("Trans=I-B"), Array("Trans=I-I")));
  
  def loadNerSystem(modelPath: String) = {
    val modelReader = IOUtils.openInHard(modelPath)
    val featureIndexer = new Indexer[String]();
    val weightsBuf = new ArrayBuffer[Double]()
    while (modelReader.ready()) {
      val line = modelReader.readLine();
      val lineSplit = line.split("\\s+");
      require(lineSplit.size == 2);
      featureIndexer.getIndex(lineSplit(0));
      weightsBuf += lineSplit(1).toDouble;
    }
    val weights = weightsBuf.toArray;
    new NerSystem(featureIndexer, weights);
  }
  
  def trainNerSystem() {
    val featureIndexer = new Indexer[String]();
    val featurizedTransitionMatrix = TransitionMatrix.map(_.map(arr => if (arr != null) arr.map(featureIndexer.getIndex(_)) else null));
    // Use the gold annotation layers: this gives gold POS (maybe not optimal?) but it's
    // obviously much better to train against gold NER than predicted...
    val useGoldAnnotation = true;
    val trainExamples = extractNerChunksFromConll(CorefSystem.loadRawConllDocs(NerDriver.trainPath, NerDriver.trainSize, useGoldAnnotation));
    val testExamples = extractNerChunksFromConll(CorefSystem.loadRawConllDocs(NerDriver.testPath, NerDriver.testSize, useGoldAnnotation));
    // Featurize examples
    val trainSequenceExs = trainExamples.map(ex => {
      new SequenceExample(featurizedTransitionMatrix, ex.featurize(featureIndexer, true), ex.goldLabels.map(LabelIndexer.getIndex(_)).toArray);
    });
    val testSequenceExs = testExamples.map(ex => {
      new SequenceExample(featurizedTransitionMatrix, ex.featurize(featureIndexer, false), ex.goldLabels.map(LabelIndexer.getIndex(_)).toArray);
    });
    val featsByType = featureIndexer.getObjects().asScala.groupBy(str => str.substring(0, str.indexOf("=")));
    Logger.logss(featureIndexer.size + " features");
    Logger.logss("Num feats each type:\n" + featsByType.map(pair => pair._1 + ": " + pair._2.size).reduce(_ + "\n" + _));
    // Train
    val weights = new Array[Double](featureIndexer.size);
    new GeneralLogisticRegression(false, false).trainWeightsLbfgsL2R(trainSequenceExs.asJava, 0.001, 0.01, 50, weights);
    var correct = 0.0;
    var predicted = 0.0;
    var actual = 0.0;
    for (trainEx <- trainSequenceExs) {
      val pred = trainEx.decode(weights).map(LabelIndexer.getObject(_));
      val gold = trainEx.goldLabels.map(LabelIndexer.getObject(_));
//      Logger.logss("PRED: " + pred.toSeq);
//      Logger.logss("GOLD: " + gold.toSeq);
      val predChunks = convertToChunks(pred);
      val goldChunks= convertToChunks(gold);
      correct += predChunks.filter(chunk => goldChunks.contains(chunk)).size;
      predicted += predChunks.size;
      actual += goldChunks.size;
    }
    Logger.logss(renderPRF1(correct, predicted, actual));
    // Decode and check test set accuracy
    correct = 0.0;
    predicted = 0.0;
    actual = 0.0;
    for (testEx <- testSequenceExs) {
      val pred = testEx.decode(weights).map(LabelIndexer.getObject(_));
      val gold = testEx.goldLabels.map(LabelIndexer.getObject(_));
//      Logger.logss("PRED: " + pred.toSeq);
//      Logger.logss("GOLD: " + gold.toSeq);
      val predChunks = convertToChunks(pred);
      val goldChunks= convertToChunks(gold);
      correct += predChunks.filter(chunk => goldChunks.contains(chunk)).size;
      predicted += predChunks.size;
      actual += goldChunks.size;
    }
    Logger.logss(renderPRF1(correct, predicted, actual));
    val modelWriter = IOUtils.openOutHard(NerDriver.modelPath);
    for (i <- 0 until featureIndexer.size) {
      modelWriter.println(featureIndexer.get(i) + " " + weights(i));
    }
    modelWriter.close();
  }
  
  
  
  private def renderPRF1(correct: Double, predicted: Double, actual: Double) = {
    val p = correct/predicted;
    val r = correct/actual;
    val f1 = 2 * p * r / (p + r);
    f1 + "; P = " + renderNumerDenom(correct, predicted) + ", R = " + renderNumerDenom(correct, actual);
  }
  
  private def renderNumerDenom(numer: Double, denom: Double) = {
    numer + "/" + denom + " = " + GUtil.fmtTwoDigitNumber(numer/denom, 2);
  }
  
  private def convertToChunks(labelSeq: Seq[String]): Seq[(Int, Int)] = {
    val chunks = new ArrayBuffer[(Int, Int)];
    for (i <- 0 until labelSeq.size) {
      if (labelSeq(i).startsWith("B")) {
        var endPoint = i + 1;
        while (endPoint < labelSeq.size && labelSeq(endPoint) != "I") {
          endPoint += 1;
        }
        chunks += i -> endPoint;
      }
    }
    chunks;
  }
  
  private def extractNerChunksFromConll(docs: Seq[ConllDoc]): Seq[NerExample] = {
    extractChunkSequences(docs, true)
  }
  
  private def extractChunkSequences(docs: Seq[ConllDoc], useNer: Boolean): Seq[NerExample] = {
    val labelIndexer = new Indexer[String]();
    labelIndexer.add("O");
    labelIndexer.add("B");
    labelIndexer.add("I");
    docs.flatMap(doc => {
      val chunksToUse = if (useNer) {
        doc.nerChunks.map(sentenceChunks => sentenceChunks.filter(chunk =>
          chunk.label != "CARDINAL" && chunk.label != "DATE" && chunk.label != "MONEY"  && chunk.label != "ORDINAL" &&
          chunk.label != "PERCENT" && chunk.label != "QUANTITY"  && chunk.label != "TIME"));
      } else {
        doc.corefChunks;
      }
      (0 until chunksToUse.size).map(sentIdx => {
        val corefChunks = chunksToUse(sentIdx);
        val labels = for (i <- 0 until doc.words(sentIdx).size) yield {
          val inChunk = corefChunks.map(chunk => if (chunk.start <= i && i < chunk.end) true else false).foldLeft(false)(_ || _);
          val startsChunk = corefChunks.map(chunk => if (chunk.start == i) true else false).foldLeft(false)(_ || _);
          if (startsChunk) "B" else if (inChunk) "I" else "O";
        }
        new NerExample(doc.words(sentIdx), doc.pos(sentIdx), labels, labelIndexer);
      })
    });
  }
}