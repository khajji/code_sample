package tweets.prediction.predictors;

import java.io.IOException;
import java.util.ArrayList;

import tweets.prediction.predictors.iface.Predictor;
import tweets.tf.main.Mixture;
import utils.DataStructures;
import utils.exceptions.IncorrectProbabilityException;
import utils.geogpraphy.GpsPoint;
import utils.geogpraphy.SubArea;
import utils.parsing.TweetParser;

/**
 * TMixturePredictor: Tweet Mixture Predictor is a predictor that uses a Mixture of Multinomials model </br>
 * to predict the class to which belong a user from the places he visited (the visited places are indicated by his tweets. </br>
 * It takes as initialization a Mixture and the name of the classes and provide methods like predict and getLikelihood
 * @author khalilhajji
 *
 */
public class TMixturePredictor implements Predictor{
	protected final String PREDICTOR_LABEL = "mixture predictor";
	private Mixture<SubArea> mixture;
	private ArrayList<SubArea> subAreas;
	
	private String[] classLabels;
	
	private double likelihood;
	
	private int samples;
	
	public TMixturePredictor(Mixture<SubArea> mixture, String[] classLabels){
		this.mixture = mixture;
		subAreas = mixture.getWordsList();
		this.classLabels = classLabels;
	}
	
	/**
	 * predicts the class of the user using the mixture model
	 * @param tweets set of tweets issued by one person
	 * @return class to which belongs the issuer of those tweets.
	 */
	@Override
	public int predict(String[] tweets){
		// TODO Auto-generated method stub
		//Note: not using the log may cause the multiplications explode
		//log(Pr(w1,..,wn|c)) = log(pr(c))+ sum<wi>(log(pr(wi|c))
		
		double[] logCKnowingTweetsArray = new double[mixture.numberOfClasses()];
		for (int i = 0; i < logCKnowingTweetsArray.length; i++) {
			logCKnowingTweetsArray[i] = Math.log(mixture.probabilityPi(i));
			
		}
		
		
		for (String tweet : tweets) {
			int subAreaId = this.getSubAreaId(tweet);
			if (subAreaId != -1) {
				this.samples++;
				for (int i = 0; i < logCKnowingTweetsArray.length; i++) {
					logCKnowingTweetsArray[i] += Math.log(mixture.probabilityWKnowingC(subAreaId, i)) ;
				}
				
			}	
		}
		
		int predictedClass = DataStructures.indexOfMax(logCKnowingTweetsArray);
		
		//compute likelihood: could do it faster but do not have performances issues, so
		//prefer to let the code simpler to understand.
		double likelihood = 0;
		for (String tweet : tweets) {
			int subAreaId = this.getSubAreaId(tweet);
			if (subAreaId != -1) {
					likelihood += Math.log(mixture.probabilityWKnowingC(subAreaId, predictedClass)) ;
				
				
			}	
		}
		
		this.likelihood += likelihood;
		
		
		return predictedClass;
	}

	@Override
	public int getNumberOfClasses() {
		// TODO Auto-generated method stub
		return mixture.numberOfClasses();
	}

	@Override
	public String[] getClassLabels() {
		// TODO Auto-generated method stub
		return this.classLabels;
	}

	@Override
	public String getLabel() {
		// TODO Auto-generated method stub
		return this.PREDICTOR_LABEL;
	}

	@Override
	public void train(String dataset) throws IOException {
		// TODO Auto-generated method stub
		//The TMixture Predictor is already trained
		
	}
	
	private int getSubAreaId(String tweet){
		TweetParser.parseTweet(tweet);
		GpsPoint tweetLocation = new GpsPoint(TweetParser.getLatitude(), TweetParser.getLongitude());
		
		boolean isFound = false;
		int index = 0;
		
		while (!isFound && index<this.subAreas.size()) {
			SubArea sub = this.subAreas.get(index);
			if (sub.contains(tweetLocation)) {
				isFound = true;
			} else {
				index++;
			}
			
		}
		
		if (index>this.subAreas.size()-1) {
			index = -1;
		}
		
		return index;
	}
	
	
	/**
	 * 
	 * @return the likelihood of the last predicted entry
	 */
	public double getLikelihood() {
		return this.likelihood/this.samples;
	}
 

}
