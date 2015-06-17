package tweets.prediction.predictors.iface;

import java.io.IOException;

/**
 * Predictor represents a predictor for tweets. Given some tweets issued by one person, decide for the 
 * </br> class of that person
 * @author khalilhajji
 *
 */
public interface Predictor {
	
	/**
	 * 
	 * @param tweets set of tweets issued by one person
	 * @return class to which belongs the user.
	 */
	public int predict(String[] tweets) throws Exception;
	
	/**
	 * 
	 * @return the number of different classes
	 */
	public int getNumberOfClasses();
	
	/**
	 * 
	 * @return the labels of the different classes
	 */
	public String[] getClassLabels();
	
	/**
	 * 
	 * @return the label of the predictor
	 */
	public String getLabel();
	
	/**
	 * train the predictor into the training dataset
	 * @param dataset the path to the training dataset
	 */
	public void train(String dataset) throws IOException;
	
	public double getLikelihood() ;

}
