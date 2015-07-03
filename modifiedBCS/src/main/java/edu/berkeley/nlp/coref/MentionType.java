package edu.berkeley.nlp.coref;


public enum MentionType {

  PROPER(false), NOMINAL(false), PRONOMINAL(true), DEMONSTRATIVE(true);
  
  private boolean isClosedClass;
  
  private MentionType(boolean isClosedClass) {
    this.isClosedClass = isClosedClass;
  }
  
  public boolean isClosedClass() {
    return isClosedClass;
  }
}
