package learning;

import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.List;

import data.Constants;
import data.Instance;
import data.Utilities;

/* IMPLEMENTED USING THE GRADIENT DESCENT ALGORITHM. */
public class LogisticRegressionGradDescent {
	
	// M: the number of the total trained data, aka total trained images
	private static int M;
	
	// N: the number features
	private static int N = Constants.NO_OF_FEATURES;
	
	private static int maxiters = 500;
	private static double alpha = 1; // small positive value, aka rate of learning
	private static double tol = 1e-6;  // tolerance

	/** the weights to train */
	private static double[] weights = new double[N+1];
	

	public static void main(String... args) throws NumberFormatException, IOException {
    	
    	// the first weight is set to 1
    	weights[0] = 1;
    	
    	/*** Train data ***/
    	// number of train spam files: 350
    	// number of train ham files: 350
    	
    	// First read the feature dictionary as a String array
    	String[] features = Utilities.readFeatureDictionary(Constants.FEATURE_DICTIONARY_PATH);
    	
    	System.out.println("Fetching train data...");
		List<Instance> spamTrains = Utilities.readData(Constants.SPAM_TRAIN_PATH, features, "train", "SPAM");
		List<Instance> hamTrains = Utilities.readData(Constants.HAM_TRAIN_PATH, features, "train", "HAM");
		List<Instance> trains = new ArrayList<Instance>();
		trains.addAll(spamTrains);
		trains.addAll(hamTrains);
		
		System.out.println();
    	System.out.println("Train data fetch completed!");
    	System.out.println();
    	
    	System.out.println("Training weights according to train data...");
		train(trains);

    	System.out.println("Training weights operation completed!");
    	System.out.println();
    	
    	/*** Test data ***/
    	// number of test spam files: 130
    	// number of test ham files: 130
    	System.out.println("Fetching test data...");
    	List<Instance> spamTests = Utilities.readData(Constants.SPAM_TEST_PATH, features, "test", "SPAM");
    	List<Instance> hamTests = Utilities.readData(Constants.HAM_TEST_PATH, features, "test", "HAM");
		List<Instance> tests = new ArrayList<Instance>();
		tests.addAll(spamTests);
		tests.addAll(hamTests);
		
    	System.out.println("Test data fetch completed!");
    	System.out.println();

    	int[] labels = new int[tests.size()];
    	for (int i=0; i<tests.size(); i++) {
    		labels[i] = tests.get(i).getLabel();
    	}
    	
    	/*** Classify the Test data ***/
    	double[] p = new double[tests.size()];

    	// Test data probabilities
    	for (int i=0; i<tests.size(); i++) {
    		
    		// vector containing the data of each row
    		int[] xtesti = tests.get(i).getX();
    		
    		p[i] = classify(xtesti);
        	
    	}

    	// from xtest0 to xtest129 -> SPAM is the correct category
    	// from xtest130 to xtest259 -> HAM is the correct category
    	
    	int wrong_counter = 0; // the number of wrong classifications made by Logistic Regression
		int spam_counter = 0; // the number of spam files
		int ham_counter = 0; // the number of ham files
		int wrong_spam_counter = 0; // the number of spam files classified as ham
		int wrong_ham_counter = 0; // the number of ham files classified as spam
    	
    	for (int i=0; i<tests.size(); i++) {
    		System.out.print("p(1|xtest" + i + "): " + p[i] + ", ");
    		
    		if (p[i] > 0.5 && labels[i] == Constants.LABEL_SPAM) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_SPAM_STRING + " -> correct");
    			spam_counter++;
    		} else if (p[i] > 0.5 && labels[i] != Constants.LABEL_SPAM) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_SPAM_STRING + " -> WRONG!");
    			ham_counter++;
    			wrong_ham_counter++;
				wrong_counter++;
    		} else if (p[i] <= 0.5 && labels[i] == Constants.LABEL_HAM) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_HAM_STRING + " -> correct");
    			ham_counter++;
    		} else if (p[i] <= 0.5 && labels[i] != Constants.LABEL_HAM) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_HAM_STRING + " -> WRONG!");
    			spam_counter++;
    			wrong_spam_counter++;
				wrong_counter++;
    		}
    		    		
    	}
    	
    	System.out.println();
    	
    	System.out.println("number of features used: " + Constants.NO_OF_FEATURES);
    	
    	System.out.println("number of wrong results: " + wrong_counter + " out of " + tests.size() + "!");
    	System.out.println("number of wrong spam classifications: " + wrong_spam_counter + " out of " + spam_counter + " spam files");
    	System.out.println("number of wrong ham classifications: " + wrong_ham_counter + " out of " + ham_counter + " ham files");
		
    	System.out.println();

		double spam_precision = (double) (spam_counter - wrong_spam_counter) / (spam_counter - wrong_spam_counter + wrong_ham_counter);
    	System.out.println("precision for spam files: " + spam_precision);
		double ham_precision = (double) (ham_counter - wrong_ham_counter) / (ham_counter - wrong_ham_counter + wrong_spam_counter);
		System.out.println("precision for ham files: " + ham_precision);

    	double spam_recall = (double) (spam_counter - wrong_spam_counter) / spam_counter;
    	System.out.println("recall for spam files: " + spam_recall);
		double ham_recall = (double) (ham_counter - wrong_ham_counter) / ham_counter;
		System.out.println("recall for ham files: " + ham_recall);
		
		double spam_f1_score = (double) 2 * spam_precision * spam_recall / (spam_precision + spam_recall);
    	System.out.println("f1-score for spam files: " + spam_f1_score);
		double ham_f1_score = (double) 2 * ham_precision * ham_recall / (ham_precision + ham_recall);
    	System.out.println("f1-score for spam files: " + ham_f1_score);
    	
    	System.out.println();

    	double accuracy = ((double) (tests.size() - wrong_counter) / tests.size()) * 100;
    	System.out.println("accuracy: " + accuracy + " %");
    	
    }
    
    
	// the gradient descent algorithm
	public static void train(List<Instance> instances) {
		
		M = instances.size();
		double J_old = Integer.MIN_VALUE;
		for (int iter=0; iter<maxiters; iter++) {
			double J = 0.0;
			for (int i=0; i<M; i++) { // for all train examples
				int[] x = instances.get(i).getX();
				double predicted = classify(x);
				int label = instances.get(i).getLabel();
				for (int j=0; j<N+1; j++) { // for all features
					double grad = (predicted - label) * x[j] / M;
					// update all the weights simultaneously
					weights[j] = weights[j] - alpha * grad;
				}
				
				// Compute the cost function.
				// We want to minimize this.
				J += - label * Math.log(predicted) - (1 - label) * Math.log(1 - predicted);
			}
					
			System.out.println("iteration: " + (iter+1) + ", cost function J: " + J);
			
			// Check for convergence.
			if (Math.abs(J - J_old) < tol) {
				break;
			}
			J_old = J;
		}
	}
	
	
	private static double classify(int[] x) {
		double logit = .0;
		for (int i=0; i<N+1; i++) {
			logit += x[i] * weights[i];
		}
		return sigmoid(logit);
	}
    
	
    // sigmoid function
    private static double sigmoid(double x) {
        return ( 1 / (1 + Math.exp(-x)) );
    }
    
}
