import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.List;


/* THIS IMPLEMENTATION USES REGULARIZATION. */
/* IMPLEMENTED USING THE GRADIENT DESCENT ALGORITHM! NO GRADIENT ASCENT! */
public class LogisticRegressionGradDescentReg {
	
	// M: the number of the total trained data, aka total trained images
	private static int M;
	
	// N: the number features, aka number of pixels in each image
	private static int N = Constants.NO_OF_PIXELS;

	private static int maxiters = 500;
	private static double alpha = 0.001; // small positive value, aka rate of learning
	private static double tol = 1e-6;  // tolerance
	private static double lambda = 0.01; // the regularization parameter
	

    public static void main(String... args) throws NumberFormatException, IOException {
    	
    	/*** Train data ***/
    	// "train1.txt" number of lines: 6742
    	// "train7.txt" number of lines: 6265
    	
    	System.out.println("Fetching train data...");
    	
		List<Instance> train1 = readDataSet(Constants.TRAIN1_PATH, Constants.LABEL_ONE);
		List<Instance> train7 = readDataSet(Constants.TRAIN7_PATH, Constants.LABEL_SEVEN);
		List<Instance> trains = new ArrayList<Instance>();
		trains.addAll(train1);
		trains.addAll(train7);
		M = trains.size();

    	System.out.println("Train data fetch completed!");
    	System.out.println();

    	System.out.println("Training weights according to train data...");
		double[] weights = train(trains);

    	System.out.println("Training weights operation completed!");
    	System.out.println();

    	/*** Test data ***/
    	// "test1.txt" number of lines: 1135
    	// "test7.txt" number of lines: 1028
    	List<Instance> test1 = readDataSet(Constants.TEST1_PATH, Constants.LABEL_ONE);
    	List<Instance> test7 = readDataSet(Constants.TEST7_PATH, Constants.LABEL_SEVEN);
		List<Instance> tests = new ArrayList<Instance>();
		tests.addAll(test1);
		tests.addAll(test7);
    	
    	/*** Classify the Test data ***/
    	double[] p = new double[tests.size()];

    	// Test data
    	for (int i=0; i<tests.size(); i++) {
    		
    		// vector containing the data of each row
    		int[] xtesti = tests.get(i).getX();
    		
    		p[i] = classify(xtesti, weights);
        	
    	}
    	
    	// from xtest0 to xtest1134 -> ONE is the correct category
    	// from xtest1135 to xtest2162 -> SEVEN is the correct category
    	
    	int wrong_counter = 0;
    	for (int i=0; i<tests.size(); i++) {
    		System.out.print("p(1|xtest" + i + "): " + p[i] + ", ");
    		
    		if (p[i] >= 0.5 && tests.get(i).getLabel() == Constants.LABEL_ONE) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_ONE_STRING + " ->  correct");
    		} else if (p[i] >= 0.5 && tests.get(i).getLabel() != Constants.LABEL_ONE) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_ONE_STRING + " -> WRONG!");
				wrong_counter++;
    		} else if (p[i] < 0.5 && tests.get(i).getLabel() == Constants.LABEL_SEVEN) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_SEVEN_STRING + " -> correct");
    		} else if (p[i] < 0.5 && tests.get(i).getLabel() != Constants.LABEL_SEVEN) {
    			System.out.println("xtest" + i + " classified as " + Constants.LABEL_SEVEN_STRING + " -> WRONG!");
				wrong_counter++;
    		}
    		
    	}
    	
    	System.out.println();
    	System.out.println("Number of wrong results: " + wrong_counter + " out of " + tests.size() + "!");
    	
    	double accuracy = ((double) (tests.size() - wrong_counter) / tests.size()) * 100;
    	System.out.println("accuracy: " + accuracy + " %");
    }
    
    
    // the gradient descent algorithm
	public static double[] train(List<Instance> instances) {

		/** the weights to train */
		double[] weights = new double[N+1];
		
		// the first weight is set to 1
    	weights[0] = 1;
    	
		M = instances.size();
		double J_old = Integer.MIN_VALUE;
		for (int iter=0; iter<maxiters; iter++) {
			double J = 0.0;
			for (int i=0; i<M; i++) { // for all train examples
				int[] x = instances.get(i).getX();
				double predicted = classify(x, weights);
				int label = instances.get(i).getLabel();
				
				// the first weight is updated differently
				double grad0 = (predicted - label) * x[0] / M;
				weights[0] = weights[0] - alpha * grad0;
				for (int j=1; j<N+1; j++) { // for all features
					double grad = (predicted - label) * x[j] / M + lambda * weights[j] / M;
					// update all the weights simultaneously
					weights[j] = weights[j] - alpha * grad ;
				}
				
				// Compute the cost function.
				// We want to minimize this.
				J += - label * Math.log(predicted) - (1 - label) * Math.log(1 - predicted) + lambda * sumOfSquares(weights) / (2*M);
			}
			
			System.out.println("iteration: " + (iter+1) + ", cost function J: " + J);
			
			// Check for convergence.
			if (Math.abs(J - J_old) < tol) {
				break;
			}
			J_old = J;
		}
		
		return weights;
	}
	
	
	// returns the sum of all elements of the given array
	private static double sumOfSquares(double[] x) {
		double sum = 0;
		// we start from index 1
		for (int i=1; i<x.length; i++) {
			sum += Math.pow(x[i], 2);
		}
		return sum;
	}
	
	
	private static double classify(int[] x, double[] weights) {
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
    
    
    // for reading Train & Test data
    private static List<Instance> readDataSet(String file, int label) {
		List<Instance> data = new ArrayList<Instance>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(file)));
		
			String line = null;
			
			while ((line = br.readLine()) != null) {
				String[] stringline = line.split(" ");
				int[] intline = new int[N+1];
				
				// the first feature is always 1
				intline[0] = 1;
				for(int i=0; i<stringline.length; i++) {
					intline[i+1] = Integer.valueOf(stringline[i]);
				}
				
				data.add(new Instance(intline, label));
			}
			br.close();
		
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return data;
	}
    
    
}
