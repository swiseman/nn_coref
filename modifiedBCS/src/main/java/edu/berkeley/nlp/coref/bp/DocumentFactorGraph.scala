package edu.berkeley.nlp.coref.bp
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.coref.Driver
import scala.util.Random
import edu.berkeley.nlp.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.coref.PairwiseScorer
import edu.berkeley.nlp.coref.GUtil
import edu.berkeley.nlp.coref.DocumentGraph
import edu.berkeley.nlp.coref.CorefDoc

class DocumentFactorGraph(val docGraph: DocumentGraph,
                          val featurizer: PairwiseIndexingFeaturizer,
                          val gold: Boolean) {
  var featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);

  val antecedentNodes = new Array[Node[Int]](docGraph.size);
//  val latentNodes = new Array[Array[Node[String]]](docGraph.size);
  val latentNodes: Array[Array[Node[String]]] = Array.tabulate(docGraph.size)(i => new Array[Node[String]](docGraph.numClusterers));
  val latentProjClusterNodes: Array[Array[Node[String]]] = Array.tabulate(docGraph.size)(i => new Array[Node[String]](docGraph.numClusterers));

  val allNodes = new ArrayBuffer[Node[_]]();
  val allNodesEveryIter = new ArrayBuffer[Node[_]]();

  val antecedentUnaryFactors = new Array[UnaryFactorOld](docGraph.size);
  val latentUnaryFactors: Array[Array[Factor]] = Array.tabulate(docGraph.size)(i => new Array[Factor](docGraph.numClusterers));

  // N.B. we don't know how big the innermost array is so don't mess with it
  val latentAgreementFactors: Array[Array[Array[Factor]]] = Array.tabulate(docGraph.size)(i => new Array[Array[Factor]](docGraph.numClusterers));
  val latentProjClusterAgreementFactors: Array[Array[AgreementFactor]] = Array.tabulate(docGraph.size)(i => new Array[AgreementFactor](docGraph.numClusterers));

  val allFactors = new ArrayBuffer[Factor]();
  val allFactorsEveryIter = new ArrayBuffer[Factor]();
  
  // LATENT
  // Notes on the clusterFeats options as they relate here: normally the projection 
  // may reduce the  number of clusters (if corefClusters is smaller than the number
  // of defined clusters), but sometimes we just want the square projection matrix for
  // clusters of different sizes. Therefore, there's also the "preserveclusterdomains"
  // option that preserves all cluster domains.
  val numClusterers = docGraph.numClusterers;
  val numLatentClustersVect = (0 until numClusterers).map(docGraph.numClusters(_));
  val numCorefClustersVect = if (Driver.clusterFeats.contains("proj") && !Driver.clusterFeats.contains("preserveclusterdomains")) {
    Array.tabulate(docGraph.numClusterers)(i => Driver.corefClusters).toSeq;
  } else {
    numLatentClustersVect;
  }
  val latentDomainsVect = numCorefClustersVect.map(numClusters => new Domain((0 until numClusters).map(_ + "").toArray));
  // First index is the cluster ID
  val latentGrid = for (cid <- 0 until numClusterers) yield {
    if (Driver.clusterFeats.contains("proj") || Driver.clusterFeats.contains("hard")) {
      DocumentFactorGraph.makeFeatureGridEmpty(latentDomainsVect(cid));
    } else if (Driver.clusterFeats.contains("agcustom")) {
      DocumentFactorGraph.makeFeatureGridAgreeCustom(latentDomainsVect(cid), DocumentFactorGraph.LatentPairwiseFeat + "C" + cid);
    } else {
      DocumentFactorGraph.makeFeatureGridHalfParameterized(latentDomainsVect(cid), DocumentFactorGraph.LatentPairwiseFeat + "C" + cid);
    }
  }
//  val latentGridsFine: Array[Array[Array[Array[Seq[String]]]]] = if (Driver.clusterFeats.contains("fine")) {
//    Array.tabulate(MentionType.values().length, MentionType.values().length)((currTypeIdx, prevTypeIdx) => {
//      DocumentFactorGraph.makeFeatureGridFineFullyParameterized(latentDomain, DocumentFactorGraph.LatentPairwiseFeat, MentionType.values()(currTypeIdx).toString(), MentionType.values()(prevTypeIdx).toString())
//    });
//  } else {
//    null;
//  }
  
//  val latentGridIndexed = latentGrid.map(_.map(_.map(featurizer.getIndex(_, false))));
  val latentGridIndexed = latentGrid.map(_.map(_.map(_.map(featurizer.getIndex(_, false)))));
  
//  val latentGridsFineIndexed = if (Driver.clusterFeats.contains("fine")) {
//    latentGridsFine.map(_.map(_.map(_.map(_.map(featurizer.getIndex(_, false))))));
//  } else {
//    null;
//  }
  val latentDefaultWeightsGrids = for (cid <- 0 until numClusterers) yield {
    if (Driver.clusterFeats.contains("proj") || Driver.clusterFeats.contains("hard")) {
      DocumentFactorGraph.makeForcedAgreementWeightsGrid(latentDomainsVect(cid).size, latentDomainsVect(cid).size);
    } else {
      DocumentFactorGraph.makeZeroWeightsGrid(latentDomainsVect(cid).size, latentDomainsVect(cid).size);
    }
  }
  
  val latentProjClusterDomainsVect = numLatentClustersVect.map(numClusters => new Domain((0 until numClusters).map(_ + "").toArray));
  val latentProjClusterGrid = for (cid <- 0 until docGraph.numClusterers) yield {
    if (Driver.clusterFeats.contains("projagree")) {
      DocumentFactorGraph.makeFeatureGridAgreeCustomOnSource(latentProjClusterDomainsVect(cid), DocumentFactorGraph.LatentProjFeat + "C" + cid);
    } else if (Driver.clusterFeats.contains("proj")) {
      DocumentFactorGraph.makeFeatureGridFullyParameterized(latentProjClusterDomainsVect(cid), latentDomainsVect(cid), DocumentFactorGraph.LatentProjFeat + "C" + cid);
    } else {
      DocumentFactorGraph.makeFeatureGridEmpty(latentProjClusterDomainsVect(cid));
    }
  }
  
  val latentProjClusterGridIndexed = latentProjClusterGrid.map(_.map(_.map(_.map(featurizer.getIndex(_, false)))));
  
  val latentProjClusterDefaultWeightsGrids = for (cid <- 0 until numClusterers) yield {
    if (Driver.clusterFeats.contains("proj")) {
      if (Driver.projDefaultWeights == "agreeheavy") {
        DocumentFactorGraph.makeAgreementWeightsGrid(latentProjClusterDomainsVect(cid).size, latentDomainsVect(cid).size, 1.0);
      } else if (Driver.projDefaultWeights == "agreelight") {
        DocumentFactorGraph.makeAgreementWeightsGrid(latentProjClusterDomainsVect(cid).size, latentDomainsVect(cid).size, 0.01);
      } else {
        DocumentFactorGraph.makeEpsilonWeightsGrid(latentProjClusterDomainsVect(cid).size, latentDomainsVect(cid).size);
      }
    } else {
      DocumentFactorGraph.makeForcedAgreementWeightsGrid(latentProjClusterDomainsVect(cid).size, latentDomainsVect(cid).size)
    }
  }
  
  
  for (i <- 0 until docGraph.size()) {
    val domainArr = docGraph.getPrunedDomain(i, gold);
    
    // NODES
    antecedentNodes(i) = new Node[Int](new Domain(domainArr));
    if (Driver.clusterFeats.contains("latent")) {
      for (cid <- 0 until docGraph.numClusterers) {
        latentNodes(i)(cid) = new Node[String](latentDomainsVect(cid));
        latentProjClusterNodes(i)(cid) = new Node[String](latentProjClusterDomainsVect(cid));
      }
    }
    allNodes += antecedentNodes(i);
    allNodesEveryIter += antecedentNodes(i);
    if (Driver.clusterFeats.contains("latent")) {
//      allNodes ++= latentNodes(i);
      for (cid <- 0 until docGraph.numClusterers) {
        allNodes += latentProjClusterNodes(i)(cid);
        allNodes += latentNodes(i)(cid);
        allNodesEveryIter += latentNodes(i)(cid);
      }
    }
    
    // UNARY FACTORS
    antecedentUnaryFactors(i) = new UnaryFactorOld(antecedentNodes(i));
    allFactors += antecedentUnaryFactors(i);
    if (Driver.clusterFeats.contains("latent")) {
      for (cid <- 0 until docGraph.numClusterers) {
        val currLatentFactor = new UnaryFactorOld(latentProjClusterNodes(i)(cid));
        currLatentFactor.setUnaryFactor(docGraph.getClusterPosteriors(cid, i))
        latentUnaryFactors(i)(cid) = currLatentFactor;
        allFactors += latentUnaryFactors(i)(cid);
      }
    }
    
    if (Driver.clusterFeats.contains("latent")) {
      for (cid <- 0 until docGraph.numClusterers) {
        latentAgreementFactors(i)(cid) = new Array[Factor](i+1);
        if (Driver.clusterFeats.contains("projfine")) {
          throw new RuntimeException("Fine features no longer supported");
//          val typeIndex = docGraph.getMention(i).mentionType.ordinal();
//          latentProjClusterAgreementFactors(i)(cid) = new AgreementFactor(latentProjClusterNodes(i)(cid), latentNodes(i)(cid), latentProjClusterGridsFine(typeIndex)(cid), latentProjClusterGridsFineIndexed(typeIndex)(cid), latentProjClusterDefaultWeightsGrid);
        } else {
          latentProjClusterAgreementFactors(i)(cid) = new AgreementFactor(latentProjClusterNodes(i)(cid), latentNodes(i)(cid), latentProjClusterGrid(cid), latentProjClusterGridIndexed(cid), latentProjClusterDefaultWeightsGrids(cid));
        }
        allFactors += latentProjClusterAgreementFactors(i)(cid);
      }
    }
    for (j <- domainArr) {
      // Don't build a factor for a guy pointing to itself
      if (j != i) {
        if (Driver.clusterFeats.contains("latent")) {
          for (cid <- 0 until docGraph.numClusterers) {
//            if (Driver.clusterFeats.contains("fine")) {
//              // Can't do projection, we require that the default weights grid is zero
//              require(!Driver.clusterFeats.contains("proj"));
//              val currMentTypeIdx = docGraph.getMention(i).mentionType.ordinal();
//              val antMentTypeIdx = docGraph.getMention(j).mentionType.ordinal();
//              latentAgreementFactors(i)(cid)(j) = new PropertyFactor(j, latentNodes(i)(cid), antecedentNodes(i), latentNodes(j)(cid), latentGridsFine(currMentTypeIdx)(antMentTypeIdx), latentGridsFineIndexed(currMentTypeIdx)(antMentTypeIdx), latentDefaultWeightsGrid);
//            } else {
            if (Driver.clusterFeats.contains("proj")) {
              latentAgreementFactors(i)(cid)(j) = new HardPropertyFactor(j, latentNodes(i)(cid), antecedentNodes(i), latentNodes(j)(cid));
            } else {
              latentAgreementFactors(i)(cid)(j) = new PropertyFactor(j, latentNodes(i)(cid), antecedentNodes(i), latentNodes(j)(cid), latentGrid(cid), latentGridIndexed(cid), latentDefaultWeightsGrids(cid));
            }
//            }
            allFactors += latentAgreementFactors(i)(cid)(j);
            allFactorsEveryIter += latentAgreementFactors(i)(cid)(j);
          }
        }
      }
    }
  }
  
  val allFeatures = allFactors.flatMap(_.getAllAssociatedFeatures()).distinct;

  // Initialize received messages at nodes
  allNodes.foreach(_.initializeReceivedMessagesUniform());

  var nodeMillis = 0L;
  var factorMillis = 0L;
  
  Logger.logss("Document factor graph instantiated: " + docGraph.size + " mentions, " + allNodes.size + " nodes (" + allNodesEveryIter.size + " every iter), " +
               allFactors.size + " factors (" + allFactorsEveryIter.size + " every iter), " + allFeatures.size + " features, <=30 of which are: " +
               allFeatures.slice(0, Math.min(30, allFeatures.size)));
  
  def setWeights(pairwiseScorer: PairwiseScorer, lossFcn: (CorefDoc, Int, Int) => Double) {
    // These scores already have -Infinity whenever something has been pruned
    val scoresChart = docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer)._2;
    // Modify the scores to incorporate softmax-margin and whether or not we're doing gold
    val antecedents = docGraph.getGoldAntecedentsUnderCurrentPruning();
    for (i <- 0 until scoresChart.size) {
      for (j <- 0 until scoresChart(i).size) {
        if (!docGraph.isPruned(i, j)) {
          if (gold) {
            // For gold, need to restrict to those in the set of antecedents
            if (!antecedents(i).contains(j)) {
              scoresChart(i)(j) = Double.NegativeInfinity;
            }
          } else {
            // For guess, need to loss-augment
            scoresChart(i)(j) += lossFcn(docGraph.corefDoc, i, j);
          }
        }
      }
      val antecedentUnaryPotential = scoresChart(i).filter((value) => !value.isNegInfinity).map(Math.exp(_));
      if (antecedentUnaryPotential.reduce(_ + _) == 0) {
        Logger.logss("Scores chart: " + scoresChart(i).toSeq);
        Logger.logss("Ant unary pot: " + i + ": " + antecedentUnaryPotential.toSeq);
        require(false);
      }
      antecedentUnaryFactors(i).setUnaryFactor(antecedentUnaryPotential);
    }
    // Update weights of the factors
    for (factor <- allFactors) {
      factor.setWeights(pairwiseScorer.weights);
    }
    // Scrub values of potentials. Can't just reset all to zero because they're
    // still linked to the received messages from the previous iteration, so the
    // arrays themselves need to be reinitialized.
//    allNodes.map(_.resetReceivedMessages());
    allNodes.foreach(_.initializeReceivedMessagesUniform());
    
    // Send initial messages from unary factors; these don't rely
    // on having received messages
    antecedentUnaryFactors.foreach(_.sendMessages());
    if (Driver.clusterFeats.contains("latent")) {
      latentUnaryFactors.foreach(_.foreach(_.sendMessages()));
    }
  }

  def passMessagesOneRound(firstOrLastIter: Boolean) {
    // Nodes and factors are ordered by position in the graph so later guys get better information from earlier ones
    val time1 = System.nanoTime();
    for (node <- if (firstOrLastIter) allNodes else allNodesEveryIter) {
      node.sendMessages();
    }
    val time2 = System.nanoTime();
    nodeMillis += (time2 - time1) / 1000000;
    for (factor <- if (firstOrLastIter) allFactors else allFactorsEveryIter) {
      factor.sendMessages();
    }
    factorMillis += (System.nanoTime() - time2) / 1000000;
  }
  
  def getDenseAntecedentNodeMarginals(idx: Int): Array[Double] = {
    val marginals = Array.fill(idx+1)(0.0);
    val sparseMarginals = antecedentNodes(idx).getMarginals();
    for (j <- 0 until sparseMarginals.size) {
      marginals(antecedentNodes(idx).domain.entries(j)) = sparseMarginals(j);
    }
    marginals;
  }

  def addExpectedFeatureCountsToGradient(scale: Double, gradient: Array[Double]) {
    val time = System.nanoTime();
    // Add pairwise features with custom machinery
    // TODO: These can be incorporated into the unary factor
    for (i <- 0 until docGraph.size) {
      val currNodeMarginals = getDenseAntecedentNodeMarginals(i);
      for (j <- 0 until currNodeMarginals.size) {
        require(currNodeMarginals(j) >= 0 && currNodeMarginals(j) <= 1);
        addToGradient(featsChart(i)(j), scale * currNodeMarginals(j), gradient);
      }
    }
    for (factor <- allFactors) {
      factor.addExpectedFeatureCounts(scale, gradient);
    }
//    Logger.logss("Marginals time: " + (System.nanoTime() - time) / 1000000 + " millis");
  }

  private def addToGradient(feats: Seq[Int], scale: Double, gradient: Array[Double]) {
    require(!scale.isNaN() && !scale.isInfinite());
    var i = 0;
    while (i < feats.size) {
      val feat = feats(i);
      gradient(feat) += 1.0 * scale;
      i += 1;
    }
  }
  
//  def computeUncertaintyStatistics(): Seq[Int] = {
//    docGraph.computeUncertaintyStatistics((idx) => antecedentNodes(idx).getMarginals());
//  }
//
//  def computeUnkStatistics(): Seq[Int] = {
//    docGraph.computeUnkStatistics();
//  }
  
  def renderLatentInfo(): String = {
    ""
//    var latentInfo = "";
//    for (i <- 0 until docGraph.size) {
//      if (antecedentNodes(i).domain.size > 1) {
//        // Only do things based on the first clusterer
//        latentInfo += i + " original: " + GUtil.fmt(docGraph.getClusterPosteriors(0, i)) + "\n"
//        latentInfo += i + " backptrs: " + antecedentNodes(i).domain.entries.toSeq + ": " + GUtil.fmt(antecedentNodes(i).getMarginals()) + "\n"
//        latentInfo += i + "    final: " + GUtil.fmt(latentNodes(i)(0).getMarginals()) + "\n";
//      } else if (antecedentNodes(i).domain.entries(0) == i) {
//        latentInfo += i + " TRIVIAL original: " + GUtil.fmt(docGraph.getClusterPosteriors(0, i)) + "\n"
//        latentInfo += i + "    TRIVIAL final: " + GUtil.fmt(latentNodes(i)(0).getMarginals()) + "\n";
//      }
//    }
//    latentInfo;
  }
}

object DocumentFactorGraph {
  val nerTypes = Seq("CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
                                  "ORDINAL", "ORG", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART");
  val nerTypesIncludingO = nerTypes ++ Seq("O");
  val nertDomain = new Domain(nerTypes.toArray);
  // Don't include "O", assume that it represents unknown
  
  val LatentPairwiseFeat = "LatentPairwise";
  val LatentProjFeat = "LatentProj";
  
  def makeUnaryFeaturesAndDefaultWeights[A](domain: Domain[A], prediction: String, agreeFeat: String, disagreeFeat: String): (Array[Seq[String]], Array[Double]) = {
    val features = new Array[Seq[String]](domain.size);
    if (Driver.clusterFeats.contains("origcustom")) {
      for (i <- 0 until domain.size) {
        features(i) = Seq("ORIGCOMPATIBLE:" + domain.entries(i) + "-p=" + prediction);
      }
    } else { // if (Driver.clusterFeats.contains("origagree")) {
      for (i <- 0 until domain.size) {
        features(i) = if (domain.entries(i).toString() == prediction) Seq(agreeFeat) else Seq(disagreeFeat);
      }
    }
    val weights = new Array[Double](domain.size);
    for (i <- 0 until domain.size) {
      weights(i) = if (Driver.clusterFeats.contains("origcustom")) {
        if (domain.entries(i).toString() == prediction) 1.0 else 0.0;
      } else {
        0.0;
      }
    }
    (features, weights);
  }
        
  def makeFeatureGrid[A](domain: Domain[A], overallName: String, agreeFeat: String, disagreeFeat: String, custom: Boolean): Array[Array[Seq[String]]] = {
    if (custom) {
      val featureGrid = new Array[Array[Seq[String]]](domain.size);
      for (i <- 0 until domain.size) {
        featureGrid(i) = new Array[Seq[String]](domain.size);
        for (j <- 0 until domain.size) {
          featureGrid(i)(j) = Seq("TRANSCOMPATIBLE-" + overallName + ":" + domain.entries(i) + "-" + domain.entries(j));
        }
      }
      featureGrid;
    } else {
      val featureGrid = new Array[Array[Seq[String]]](domain.size);
      for (i <- 0 until domain.size) {
        featureGrid(i) = new Array[Seq[String]](domain.size);
        for (j <- 0 until domain.size) {
          featureGrid(i)(j) = if (i == j) Seq(agreeFeat) else Seq(disagreeFeat);
        }
      }
      featureGrid;
    }
  }
        
  def makeFeatureGridAgreeCustom[A](domain: Domain[A], featPrefix: String): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domain.size);
    for (i <- 0 until domain.size) {
      featureGrid(i) = new Array[Seq[String]](domain.size);
      for (j <- 0 until domain.size) {
        featureGrid(i)(j) = Seq(if (i == j) featPrefix + "Agree-" + i else featPrefix + "Disagree");
      }
    }
    featureGrid;
  }
        
  def makeFeatureGridAgreeCustomOnSource[A](domain: Domain[A], featPrefix: String): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domain.size);
    for (i <- 0 until domain.size) {
      featureGrid(i) = new Array[Seq[String]](domain.size);
      for (j <- 0 until domain.size) {
        featureGrid(i)(j) = Seq(if (i == j) featPrefix + "Agree-" + i else featPrefix + "Disagree-" + i);
      }
    }
    featureGrid;
  }
  
  def makeFeatureGridHalfParameterized[A](domain: Domain[A], featPrefix: String): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domain.size);
    for (i <- 0 until domain.size) {
      featureGrid(i) = new Array[Seq[String]](domain.size);
      for (j <- 0 until domain.size) {
        featureGrid(i)(j) = Seq(featPrefix + "-" + (if (i <= j) i + "-" + j else j + "-" + i));
      }
    }
    featureGrid;
  }
        
  def makeFeatureGridFullyParameterized[A](domainOne: Domain[A], domainTwo: Domain[A], featPrefix: String): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domainOne.size);
    for (i <- 0 until domainOne.size) {
      featureGrid(i) = new Array[Seq[String]](domainTwo.size);
      for (j <- 0 until domainTwo.size) {
        featureGrid(i)(j) = Seq(featPrefix + "-" + i + "-" + j);
      }
    }
    featureGrid;
  }
  
  def makeFeatureGridFineFullyParameterized[A](domain: Domain[A], featPrefix: String, conjCurr: String, conjPrev: String): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domain.size);
    for (i <- 0 until domain.size) {
      featureGrid(i) = new Array[Seq[String]](domain.size);
      for (j <- 0 until domain.size) {
        featureGrid(i)(j) = Seq(featPrefix + "-" + i + "-" + j, featPrefix + "-" + i + "-" + j + "&" + conjCurr, featPrefix + "-" + i + "-" + j + "&" + conjCurr + "&" + conjPrev);
      }
    }
    featureGrid;
  }
        
  def makeFeatureGridEmpty[A](domain: Domain[A]): Array[Array[Seq[String]]] = {
    val featureGrid = new Array[Array[Seq[String]]](domain.size);
    for (i <- 0 until domain.size) {
      featureGrid(i) = new Array[Seq[String]](domain.size);
      for (j <- 0 until domain.size) {
        featureGrid(i)(j) = Seq[String]();
      }
    }
    featureGrid;
  }
  
//  def makeFeatureGridFullyParameterizedConjunctions[A](domain: Domain[A], agreeFeat: String, currMentType: MentionType, antMentType: MentionType): Array[Array[Seq[String]]] = {
//    val featureGrid = new Array[Array[Seq[String]]](domain.size);
//    for (i <- 0 until domain.size) {
//      featureGrid(i) = new Array[Seq[String]](domain.size);
//      for (j <- 0 until domain.size) {
//        featureGrid(i)(j) = Seq(agreeFeat + (if (i <= j) i + "-" + j else j + "-" + i));
//        featureGrid(i)(j) = Seq(agreeFeat + i + "-" + j,
//                                agreeFeat + i + "-" + j + "-Curr=" + currMentType.toString,
//                                agreeFeat + i + "-" + j + "-Curr=" + currMentType.toString + "-Prev=" + antMentType.toString);
//      }
//    }
//    featureGrid;
//  }
  
  def makeZeroWeightsGrid(dim1: Int, dim2: Int): Array[Array[Double]] = {
    Array.tabulate(dim1, dim2)((i, j) => 0.0);
  }
  
  def makeForcedAgreementWeightsGrid(dim1: Int, dim2: Int): Array[Array[Double]] = {
    require(dim1 == dim2, "Can only force agreement on square matrix");
    Array.tabulate(dim1, dim2)((i, j) => if (i == j) 0.0 else Double.NegativeInfinity);
  }
  
  def makeAgreementWeightsGrid(dim1: Int, dim2: Int, agreementWeight: Double): Array[Array[Double]] = {
    require(dim1 == dim2, "Can only have agreement on square matrix");
    Array.tabulate(dim1, dim2)((i, j) => if (i == j) agreementWeight else 0.0);
  }
  
  def makeEpsilonWeightsGrid(dim1: Int, dim2: Int): Array[Array[Double]] = {
    val rand = new Random(0);
    Array.tabulate(dim1, dim2)((i, j) => (rand.nextDouble() - 0.5) * 0.01);
  }
}
