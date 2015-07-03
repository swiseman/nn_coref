package edu.berkeley.nlp.coref

case class Feature(context: String, event: String, value: Double, basic: Boolean) {
  val name = context + " >> " + event;
  val contextAndTemplate = context + ":" + (if (basic) "basic" else "conj");
};
