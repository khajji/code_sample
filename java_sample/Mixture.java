package tweets.tf.main;

import java.util.ArrayList;
import java.util.HashMap;
import utils.Utils;
import utils.exceptions.IncorrectProbabilityException;

/**
 * 
 * @author khalilhajji
 * Mixture is a class that models a generative mixture of words (w) in a corpus of documents (or classes c). </br>
 * the generative model works as following: </br>
 * Pr(w) = Sum<c>(Pr(c). Sum<z>(Pr(z|c).Pr(w|z))) where z are hidden latent variables.</br>
 * 
 * This class is instantiated by the described probabilities Pr(c), Pr(z|c) and Pr(w|z) </br>
 * and provide methods to compute Pr(c|w), Pr(w|c), Pr(z|w), Pr(w) in addition to the existing distributions.
 * 
 * @param <T> The Type of words of the Mixture. A mixture do not models only a set of Strings(Words) in a corpus of an ensemble of Strings (Doc). </br>
 * Words represent a geographical sub-area inside a geographic area. It can represent any feature in a set of features that describe a certain behaviour. </br>
 * We want our class to be as generic as possible to be able to represent all those kind of mixtures. 
 */
public class Mixture<T> {
	
	private ArrayList<T> words;
	private HashMap<Integer, ArrayList<Double>> wKonwingZ;
	private HashMap<Integer, ArrayList<Double>> zKnowingC;
	private ArrayList<Double> pi;

	private ArrayList<Double> w;
	private HashMap<Integer, ArrayList<Double>> zKnowingW;
	private HashMap<Integer, ArrayList<Double>> cKnowingW;

	private int zSize;


	/**
	 * we will call words w, latent topics z and classes c
	 * @param words the list of words (ordered so that ids match with Pr(w|z)
	 * @param wKonwingZ Pr(w|z) distribution where each key zk contains an arraylist of the size of the vocabulary representing Pr(w|zk) distribution for each word
	 * @param zKnowingC Pr(z|c) distribution where each key ci contains an arraylist of the size of the latent variables representing Pr(z|ci) distribution for each latent variable
	 * @param pi Pr(c) distribution
	 * @throws IncorrectProbabilityException 
	 */
	public Mixture(ArrayList<T> words, HashMap<Integer, ArrayList<Double>> wKonwingZ, HashMap<Integer, ArrayList<Double>> zKnowingC, ArrayList<Double> pi) throws IncorrectProbabilityException {
		this.words = words;
		this.wKonwingZ = wKonwingZ;
		this.zKnowingC = zKnowingC;
		this.pi = pi;

		this.zSize = this.wKonwingZ.size();

		this.w = this.computeW(wKonwingZ, zKnowingC, pi);
		this.zKnowingW = this.computeZKnowingW(wKonwingZ, zKnowingC, pi, w);
		this.cKnowingW = this.computeCKnowingW(wKonwingZ, zKnowingC, pi, w);

		this.sanityTest();
		System.out.println("sanity completed");




	}

	/**
	 * 
	 * @return the list of words
	 */
	public ArrayList<T> getWordsList(){
		return this.words;
	}

	/**
	 * 
	 * @param wordId must be between 1 and numberOfWords-1
	 * @param topicId must be between 1 and numberOfTopics-1
	 * @return pr(w|z)
	 */
	public double probabilityWKnowingZ(int wordId, int topicId){

		return this.wKonwingZ.get(topicId).get(wordId);


	}

	/**
	 * 
	 * @param topicId must be between 1 and numberOfTopics-1
	 * @param classId must be between 1 and numberOfClasses-1
	 * @return pr(z|c)
	 */
	public double probabilityZKnowingC(int topicId, int classId){
		return this.zKnowingC.get(classId).get(topicId);
	}

	/**
	 * 
	 * @param classId must be between 1 and numberOfClasses-1
	 * @return Pr(c) called PiC
	 */
	public double probabilityPi(int classId){
		return this.pi.get(classId);
	}

	/**
	 * 
	 * @param topicId must be between 1 and numberOfTopics-1
	 * @param wordId must be between 1 and numberOfWords-1
	 * @return Pr(z|w)
	 */
	public double probabilityZKnowingW(int topicId, int wordId){

		return this.zKnowingW.get(wordId).get(topicId);



	}

	/**
	 * 
	 * @param classId must be between 1 and numberOfClasses-1
	 * @param wordId must be between 1 and numberOfWords-1
	 * @return Pr(c|w)
	 */
	public double probabilityCKnowingW(int classId, int wordId) {

		return this.cKnowingW.get(wordId).get(classId);
	}
	
	/**
	 * 
	 * @param wordId must be between 1 and numberOfWords-1
	 * @return Pr(w)
	 */
	public double probabilityW(int wordId){

		return this.w.get(wordId);


	}
	
	/**
	 *  * @param wordId must be between 1 and numberOfWords-1
	 * @param classId must be between 1 and numberOfClasses-1
	 * @return Pr(w|c)
	 */
	public double probabilityWKnowingC(int wordId, int classId) {
		//Pr(w|c) = sum<z> Pr(z|c)*Pr(w|z)
		double result = 0;
		for (int z = 0; z < numberOfTopics(); z++) {
			result += this.probabilityZKnowingC(z, classId)*this.probabilityWKnowingZ(wordId, z);
		}


		return result;



	}

	/**
	 * 
	 * @return the total number of words
	 */
	public int numberOfWords(){
		return this.w.size();
	}

	/**
	 * 
	 * @return the total number of topics (or latent variables)
	 */
	public int numberOfTopics(){
		return this.zSize;
	}

	/**
	 * 
	 * @return the total number of classes
	 */
	public int numberOfClasses(){
		return this.pi.size();
	}

	private ArrayList<Double> computeW(HashMap<Integer, ArrayList<Double>> wKonwingZ, HashMap<Integer, ArrayList<Double>> zKnowingC, ArrayList<Double> pi){
		int wSize = wKonwingZ.get(0).size();
		int zSize = this.zSize;
		ArrayList<Double> w = new ArrayList<Double>();

		for (int wId = 0; wId < wSize; wId++) {
			//for each word, compute Pr(w) = Sum<c>Pr(c)*(Sum<z> Pr(z|c)Pr(w|z))
			double prW = 0;
			for (int cId = 0; cId < pi.size(); cId++) {
				//iterate over classes c
				double piC = pi.get(cId);
				double wKc = 0;
				for (int zId = 0; zId <zSize ; zId++) {
					wKc += wKonwingZ.get(zId).get(wId)*zKnowingC.get(cId).get(zId);
				}

				prW = prW+ (wKc*piC);


			}
			
			w.add(prW);

		}
		return w;
	}

	private HashMap<Integer, ArrayList<Double>> computeZKnowingW(HashMap<Integer, ArrayList<Double>> wKonwingZ, HashMap<Integer, ArrayList<Double>> zKnowingC, ArrayList<Double> pi, ArrayList<Double> w){
		HashMap<Integer, ArrayList<Double>> zKnowingW = new HashMap<Integer, ArrayList<Double>>();

		int wSize = w.size();
		int zSize = this.zSize;
		int cSize = pi.size();

		for (int wId = 0; wId < wSize; wId++) {
			//for each word w
			double prW = w.get(wId);
			ArrayList<Double> zKwArray = new ArrayList<Double>();
			for (int zId = 0; zId < zSize; zId++) {
				//for each word w and topic z, compute Pr(z|w) = Pr(w|z)*(Sum<c> Pr(z|c)Pr(c))/pr(w)
				double zKw = 0;
				double wKz = wKonwingZ.get(zId).get(wId);

				for (int cId = 0; cId < cSize; cId++) {
					zKw += zKnowingC.get(cId).get(zId)*pi.get(cId);
				}
				
				zKw = zKw*wKz/prW;
				
				zKwArray.add(zKw);

			}
			zKnowingW.put(wId, zKwArray);

		}
		return zKnowingW;
	}

	private HashMap<Integer,ArrayList<Double>> computeCKnowingW(HashMap<Integer, ArrayList<Double>> wKonwingZ, HashMap<Integer, ArrayList<Double>> zKnowingC, ArrayList<Double> pi, ArrayList<Double> w){
		HashMap<Integer, ArrayList<Double>> cKnowingW = new HashMap<Integer, ArrayList<Double>>();

		int wSize = w.size();
		int zSize = this.zSize;
		int cSize = pi.size();

		for (int wId = 0; wId < wSize; wId++) {
			//for each word w
			double prW = w.get(wId);
			ArrayList<Double> cKwArray = new ArrayList<Double>();
			for (int cId = 0; cId < cSize; cId++) {
				//for each word w and class c, compute Pr(c|w) = Pr(c)*(Sum<z> Pr(z|c)Pr(w|z))/pr(w)
				double cKw = 0;
				double piC = pi.get(cId);

				for (int zId = 0; zId < zSize; zId++) {
					cKw += wKonwingZ.get(zId).get(wId)*zKnowingC.get(cId).get(zId);
				}

				cKw = cKw*piC/prW;
				
				cKwArray.add(cKw);


			}
			cKnowingW.put(wId, cKwArray);

		}

		return cKnowingW;
	}


	public HashMap<Integer, ArrayList<Double>> getwKonwingZ() {
		return (HashMap<Integer, ArrayList<Double>>) wKonwingZ.clone();
	}

	public HashMap<Integer, ArrayList<Double>> getzKnowingC() {
		return (HashMap<Integer, ArrayList<Double>>) zKnowingC.clone();
	}

	public ArrayList<Double> getW() {
		return (ArrayList<Double>) w.clone();
	}

	public HashMap<Integer, ArrayList<Double>> getzKnowingW() {
		return (HashMap<Integer, ArrayList<Double>>) zKnowingW.clone();
	}

	public HashMap<Integer, ArrayList<Double>> getcKnowingW() {
		return (HashMap<Integer, ArrayList<Double>>) cKnowingW.clone();
	}

	/**
	 * ensures that the properties of the probabilities are ensured
	 * Ex: All probabilities sum to 1
	 * @throws IncorrectProbabilityException 
	 */
	private void sanityTest() throws IncorrectProbabilityException {
		//test that pr(c) sum to 1 for all c
		Utils.testDistributionSum(this.pi);


		double sum = 0;
		//test that pr(z|c) sums to 1 for all z and c
		for (int c = 0; c < numberOfClasses(); c++) {
			for (int z = 0; z < numberOfTopics(); z++) {
				sum += this.probabilityZKnowingC(z, c);

			}
			Utils.testDistributionSum(sum);
			sum = 0;

		}

		//test that pr(w|z) sums to 1 for all w and z
		sum = 0;
		for (int z = 0; z < numberOfTopics(); z++) {
			for (int w = 0; w < numberOfWords(); w++) {
				sum += probabilityWKnowingZ(w, z);

			}

			Utils.testDistributionSum(sum);
			sum = 0;


		}

		//test that pr(w|c) sums to 1 for all w and c
		sum = 0;
		for (int c = 0; c < numberOfClasses(); c++) {
			for (int w = 0; w < numberOfWords(); w++) {
				sum += probabilityWKnowingC(w, c);
			}

			Utils.testDistributionSum(sum);
			sum = 0;

		}

		//test that pr(w) sums to 1 for all w
		Utils.testDistributionSum(this.w);




	}
}
