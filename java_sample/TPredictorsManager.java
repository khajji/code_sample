package tweets.prediction.evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import tweets.prediction.predictors.iface.Predictor;
import utils.DataStructures;
import utils.exceptions.IncorrectProbabilityException;
import utils.geometry.Polygon;
import utils.parsing.DatasetLineParser;
import utils.parsing.UserParser;

/**
 * TPredictorsManager is a class that evaluates different predictors that are accomplishing the same </br>
 * prediction on the same data.
 * @author khalilhajji
 *
 */
public class TPredictorsManager {
	protected ArrayList<Predictor> predictors;
	protected ArrayList<PredictorTester> testers;
	protected Polygon area;
	protected int k;
	protected String[] classLabels;

	/**
	 * the predictors must be not trained
	 * @param predictors the predictors to evaluate
	 */
	public TPredictorsManager(ArrayList<Predictor> predictors, Polygon area) {

		this.k = predictors.get(0).getNumberOfClasses();
		this.classLabels = predictors.get(0).getClassLabels();

		this.testers = new ArrayList<PredictorTester>();
		for (int i = 0; i < predictors.size(); i++) {
			if (predictors.get(i).getNumberOfClasses() != this.k || !predictors.get(i).getClassLabels().equals(this.classLabels)) {
				throw new IllegalArgumentException("The predictors must have the same number of classes and the same class labels");
			}
			testers.add(new PredictorTester(this.k, classLabels));
		}
		this.predictors = predictors;
	}

	public TPredictorsManager(Polygon area) {
		this.predictors = new ArrayList<Predictor>();
		this.testers = new ArrayList<PredictorTester>();
	}

	/**
	 * add a predictor to the predictor manager.
	 * </br> the predictor must be not trained.
	 * @param predictor the predictor to add
	 */
	public void subscribePredictor(Predictor predictor) {
		if (this.predictors.size() == 0) {
			//it's the first predictor so initialize some variables
			this.k = predictor.getNumberOfClasses();
			this.classLabels = predictor.getClassLabels();
			this.testers.add(new PredictorTester(this.k, this.classLabels));
			this.predictors.add(predictor);
		} else {
			//check that the arguments are compatible
			if (predictor.getNumberOfClasses() != this.k || !Arrays.equals(predictor.getClassLabels(), this.classLabels)) {

				throw new IllegalArgumentException("The predictor must have "+this.k+" classes and the same class labels than the already exisisting once");
			} else {
				this.predictors.add(predictor);
				this.testers.add(new PredictorTester(this.k, this.classLabels));
			}

		}

	}

	/**
	 * asks the different predictors to predict the category of user given it's the tweets. </br>
	 * Note that the ground truth is given to the manager to enable him to evaluate the
	 * </br> predictors
	 * @param tweets the tweets must belong to the same user
	 * @param groundTruth the real class of the user
	 * @throws Exception 
	 */
	private void predict(String[] tweets, int groundTruth, String line) throws IncorrectProbabilityException, Exception {

		int index = 0;
		if (tweets.length != 0) {
			for (Predictor predictor : this.predictors) {
				int result = predictor.predict(tweets);
				this.testers.get(index).updateScore(result, groundTruth, line );
				index++;
			}	
		}
		

	}

	/**
	 * do the training of the different predictors
	 * @param trainSetFile
	 * @throws IOException
	 */
	public void trainPredictors(String trainSetFile) throws IOException {
		for (Predictor predictor : this.predictors) {
			System.out.println("training of predictor : "+predictor.getLabel());
			predictor.train(trainSetFile);
			System.out.println("end training "+predictor.getLabel());

			System.out.println();
		}

	}

	/**
	 * run the predictors into the test set
	 * @param testSetFile
	 * @throws IOException 
	 */
	public void testPredictors(String testSetFile) throws IncorrectProbabilityException, IOException, Exception {
		BufferedReader reader = new BufferedReader(new FileReader(testSetFile));

		String line = reader.readLine();
		while (line != null) {
			int userClass = getClass(line);
			if (userClass != -1) {
				DatasetLineParser.parseDatasetLine(line);
				this.predict(DatasetLineParser.getTweets(), userClass, line);

			}
			
			line = reader.readLine();

		}

	}

	public String reportPredictorsPerformance() {
		String performancesReport= "";
		for (int i = 0; i < this.predictors.size(); i++) {
			performancesReport = performancesReport +
					this.predictors.get(i).getLabel()+"\n"+
					this.testers.get(i).reportPredictorPerformances()+"\n"+
					"loglikelihood : "+this.predictors.get(i).getLikelihood()+
					"\n\n";

		}

		return performancesReport;
	}
	
	public void reportStatistics(String statfile) throws IOException{
		for (int i = 0; i < this.predictors.size(); i++) {
			this.testers.get(i).getErrorVsEthCertitude().writeStatistics(statfile+predictors.get(i).getLabel()+"errorVs#ethCertitude");
			this.testers.get(i).getErrorVsNbTweets().writeStatistics(statfile+predictors.get(i).getLabel()+"errorVs#tweets");

		}
	}

	/**
	 * 
	 * @param line one dataset line
	 * @return the class to which belong the user
	 */

	protected int getClass(String line) {
		DatasetLineParser.parseDatasetLine(line);
		UserParser.parseUser(DatasetLineParser.getUser());
		int category = DataStructures.indexOf(this.classLabels, UserParser.getCategory());
		if (category>-1) {
			return category;
		} else {
			throw new IllegalArgumentException("Category is unknown : "+category);
		}
	}

}
