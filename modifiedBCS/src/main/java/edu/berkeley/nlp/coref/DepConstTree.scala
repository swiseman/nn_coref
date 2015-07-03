package edu.berkeley.nlp.coref
import java.util.IdentityHashMap

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.JavaConverters.collectionAsScalaIterableConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import edu.berkeley.nlp.futile.fig.basic.Pair
import edu.berkeley.nlp.futile.ling.HeadFinder
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import edu.berkeley.nlp.futile.syntax.Tree
import edu.berkeley.nlp.futile.util.Logger

class DepConstTree(val constTree: Tree[String],
                   val pos: Seq[String],
                   val words: Seq[String],
                   val childParentDepMap: HashMap[Int,Int]) {
  require(childParentDepMap.keys.toSeq.sorted.sameElements((0 until words.size)), PennTreeRenderer.render(constTree));
  
  override def toString() = {
    var strRep = "";
    strRep += PennTreeRenderer.render(constTree) + "\n";
    for (i <- 0 until words.size) {
      val headIdx = childParentDepMap(i);
      strRep += words(i) + "(" + i + "), head = " + (if (headIdx == -1) "ROOT" else words(headIdx) + "(" + headIdx + ")") + "\n";
    }
    strRep;
  }
  
  /**
   * Fetches the head for an arbitrary span; this is needed for head-finding of mentions
   * that might not be constituents.
   */
  def getSpanHead(startIdx: Int, endIdx: Int) = {
    // If it's a constituent, only one should have a head outside
    val outsidePointing = new ArrayBuffer[Int];
    for (i <- startIdx until endIdx) {
      val ptr = childParentDepMap(i);
      if (ptr < startIdx || ptr >= endIdx) {
        outsidePointing += i;
      }
    }
    var isConstituent = false;
    val constituents = constTree.getConstituentCollection().asScala;
    for (const <- constituents) {
      if (const.getStart == startIdx && const.getEnd+1 == endIdx) {
        isConstituent = true;
      }
    }
//    if (isConstituent && outsidePointing.size > 1) {
//      Logger.logss("Contradiction: isConstituent but >1 ptr");
//    }
//    if (!isConstituent && outsidePointing.size == 1) {
//      Logger.logss("Contradiction: !isConstituent but ==1 ptr");
//    }
//    if (outsidePointing.size > 1) {
//      Logger.logss("TOO MANY POINTERS");
//    }
    // If our heuristic failed to identify anything, assume head final
    if (outsidePointing.isEmpty) {
      Logger.logss("WARNING: Empty outside pointing " + startIdx + ", " + endIdx + ": " + childParentDepMap);
      endIdx - 1;
    } else {
      outsidePointing.last;
    }
  }
  
  def isConstituent(start: Int, end: Int) = {
    val spans = constTree.getSpanMap();
    spans.containsKey(Pair.makePair(new Integer(start), new Integer(end)));
  }
  
  // XXX: This is broken in some subtle way
//  def cCommand(commanderStart: Int, commanderEnd: Int, commandeeStart: Int, commandeeEnd: Int): String = {
//    val spans = constTree.getSpanMap();
//    if (spans.containsKey(fig.basic.Pair.makePair(new Integer(commanderStart), new Integer(commanderEnd - 1)))) {
//      "UNKNOWN"
//    } else {
//      // Find the smallest span properly containing this one
//      var parentStart = -1;
//      var parentEnd = constTree.size() + 1;
//      for (span <- spans.keySet.asScala) {
//        val thisStart = span.getFirst.intValue;
//        val thisEnd = span.getSecond.intValue + 1;
//        val containsProperly = thisStart <= commanderStart && commanderEnd <= thisEnd && (thisStart != commanderStart || commanderEnd != thisEnd);
//        if (containsProperly && thisStart >= parentStart && thisEnd <= parentEnd) {
//          parentStart = thisStart;
//          parentEnd = thisEnd;
//        }
//      }
//      require(parentStart != -1 && parentEnd != constTree.size() + 1);
//      if (parentStart <= commandeeStart && commandeeEnd <= parentEnd) {
//        "TRUE";
//      } else {
//        "FALSE";
//      }
//    }
//  }
  
  def getSpansAndHeadsOfType(constituentType: String): Seq[(Int, Int, Int)] = {
    val results = new ArrayBuffer[(Int, Int, Int)];
    for (constituent <- constTree.getConstituentCollection().asScala) {
      if (constituent.getLabel() == constituentType) {
        results += new Tuple3(constituent.getStart(), constituent.getEnd() + 1, getSpanHead(constituent.getStart(), constituent.getEnd() + 1));
      }
    }
    results;
  }
  
  def computeSyntacticUnigram(headIdx: Int): String = {
    val parentIdx: Int = if (childParentDepMap.contains(headIdx)) childParentDepMap(headIdx) else -1;
    val parentStr = if (parentIdx == -1) {
      "NULL";
    } else {
      pos(parentIdx) + "(" + (if (headIdx > parentIdx) "L" else "R") + ")"; 
    }
    parentStr;
  }
  
  def computeSyntacticBigram(headIdx: Int): String = {
    val parentIdx: Int = if (childParentDepMap.contains(headIdx)) childParentDepMap(headIdx) else -1;
    val grandparentIdx: Int = if (parentIdx != -1 && childParentDepMap.contains(parentIdx)) childParentDepMap(parentIdx) else -1;
    val parentStr = if (parentIdx == -1) {
      "NULL";
    } else {
      pos(parentIdx) + "(" + (if (headIdx > parentIdx) "L" else "R") + ")"; 
    }
    val grandparentStr = if (grandparentIdx == -1) {
      "NULL";
    } else {
      pos(grandparentIdx) + "(" + (if (parentIdx > grandparentIdx) "L" else "R") + ")"; 
    }
    parentStr + "-" + grandparentStr;
  }
  
  def computeSyntacticPositionSimple(headIdx: Int): String = {
    val parentIdx: Int = if (childParentDepMap.contains(headIdx)) childParentDepMap(headIdx) else -1;
    val grandparentIdx: Int = if (parentIdx != -1 && childParentDepMap.contains(parentIdx)) childParentDepMap(parentIdx) else -1;
    if (parentIdx != -1 && pos(parentIdx).startsWith("V") && headIdx < parentIdx) {
      "SUBJECT";
    } else if (parentIdx != -1 && pos(parentIdx).startsWith("V") && headIdx > parentIdx) {
      "DIROBJ";
    } else if (parentIdx != -1 && grandparentIdx != -1 && (pos(parentIdx) == "IN" || pos(parentIdx) == "TO") && pos(grandparentIdx).startsWith("V")) {
      "INDIROBJ";
    } else {
      "OTHER";
    }
  }
}

object DepConstTree {
  
  def extractDependencyStructure(constTree: Tree[String], headFinder: HeadFinder): HashMap[Int, Int] = {
    // Type created by this method is an IdentityHashMap, which is correct
    // N.B. The constituent end index is the last word of the mention, it's not on fenceposts
    val constituents = constTree.getConstituents()
    val subtreeHeads = new IdentityHashMap[Tree[String],Int];
    val trees = constTree.getPostOrderTraversal().asScala;
    require(trees.last eq constTree);
    val heads = new HashMap[Int,Int]();
    for (tree <- trees) {
      if (tree.isLeaf) {
        // Do nothing
      } else if (tree.isPreTerminal) {
        val constituent = constituents.get(tree);
        require(!subtreeHeads.containsKey(tree));
        subtreeHeads.put(tree, constituent.getStart());
      } else {
        val children = tree.getChildren();
        val head = headFinder.determineHead(tree);
        if (head == null) {
          Logger.logss("WARNING: null head: " + PennTreeRenderer.render(constTree) + "\n" + PennTreeRenderer.render(tree));
        }
        val headIdx = subtreeHeads.get(head);
        for (child <- children.asScala) {
          if (child eq head) {
            subtreeHeads.put(tree, headIdx);
          } else {
            require(!heads.contains(subtreeHeads.get(child)), "\n" + PennTreeRenderer.render(constTree) +
                      "\n" + PennTreeRenderer.render(tree) +
                      "\n" + PennTreeRenderer.render(child) +
                      "\n" + heads);
            heads(subtreeHeads.get(child)) = headIdx;
          }
        }
      }
    }
    // Set the root head
    heads(subtreeHeads.get(constTree)) = -1;
    val numLeaves = constTree.getYield.size();
    for (i <- 0 until numLeaves) {
      require(heads.contains(i), heads + "\n" + PennTreeRenderer.render(constTree));
    }
    heads;
  }
  
}
