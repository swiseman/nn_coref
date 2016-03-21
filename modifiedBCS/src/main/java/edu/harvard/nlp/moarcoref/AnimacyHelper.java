package edu.harvard.nlp.moarcoref;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import edu.berkeley.nlp.coref.Mention;
import edu.berkeley.nlp.coref.MentionType;

public class AnimacyHelper {
	
  public static Set<String> animates;
  public static Set<String> inanimates;
	  
	static {
			try {
				animates = getWordsFromFile(MiniDriver.animacyPath, false);
				inanimates = getWordsFromFile(MiniDriver.inanimacyPath, false);
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(1);
			}
				
	}
	
	
	//////////////////////////////////////////////////////
	// implementation of some recasens features
	/////////////////////////////////////////////////////
	
	public static String getAnimacy(Mention ment) {
		String animacy = "UNKNOWN";
		String headString = ment.headString();
		String nerString = ment.nerString();
		Set<String> inanimateNers = new HashSet<String>(Arrays.asList(
				"LOCATION", "MONEY", "NUMBER", "PERCENT", "DATE", "TIME",
				"FAC", "GPE", "WEA", "ORG"));
		if (ment.mentionType() == MentionType.PRONOMINAL) {
			if (animates.contains(headString)) {
				animacy = "ANIMATE";
			} else if (inanimates.contains(headString)) {
				animacy = "INANIMATE";
			}
		} else if (nerString.equals("PERSON") || nerString.startsWith("PER")) {
			animacy = "ANIMATE";
		} else if (inanimateNers.contains(nerString)
				|| nerString.startsWith("LOC")) {
			animacy = "INANIMATE";
		}
		// if still unknown, use list
		if (ment.mentionType() != MentionType.PRONOMINAL
				&& animacy.equals("UNKNOWN")) {
			if (animates.contains(headString)) {
				animacy = "ANIMATE";
			} else if (inanimates.contains(headString)) {
				animacy = "INANIMATE";
			}
		}
		return animacy;
	}	
	
	// mostly stolen from Dictionaries.java in stanfordcorenlp.dcoref
	public static Set<String> getWordsFromFile(String filename, boolean lowercase) throws IOException{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
		Set<String> words = new HashSet<String>();
		while (reader.ready()){
			if (lowercase){
				words.add(reader.readLine().toLowerCase()); // readLine strips the trailing '\n' etc
			} else {
				words.add(reader.readLine());
			}
		}
		reader.close();
		return words;
	}	
}