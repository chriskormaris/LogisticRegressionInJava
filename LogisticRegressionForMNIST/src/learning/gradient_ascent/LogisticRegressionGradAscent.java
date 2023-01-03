package learning.gradient_ascent;

import data.Constants;
import data.Instance;
import data.Utilities;

import java.util.ArrayList;
import java.util.List;


/* IMPLEMENTED USING THE GRADIENT ASCENT ALGORITHM. */
public class LogisticRegressionGradAscent {

	// M: the number of the total trained data, aka total trained images
	private static int M;

	// N: the number features, aka number of pixels in each image
	private static final int N = Constants.NUM_PIXELS;

	private static final int maxiters = 500;
	private static final double alpha = 0.001; // small positive value, aka rate of learning
	private static final double tol = 1e-6;  // tolerance


	public static void main(String... args) throws NumberFormatException {

		/* Train data */
		// "train1.txt" number of lines/images: 6742
		// "train7.txt" number of lines/images: 6265

		System.out.println("Fetching train data...");

		List<Instance> train1 = Utilities.readDataSet(Constants.TRAIN1_PATH, Constants.LABEL_ONE);
		List<Instance> train7 = Utilities.readDataSet(Constants.TRAIN7_PATH, Constants.LABEL_SEVEN);
		List<Instance> trains = new ArrayList<>();
		trains.addAll(train1);
		trains.addAll(train7);
		M = trains.size();

		System.out.println("Train data fetch completed!");
		System.out.println();

		System.out.println("Training weights according to train data...");
		double[] weights = train(trains);

		System.out.println("Training weights operation completed!");
		System.out.println();

		/* Test data */
		// "test1.txt" number of lines/images: 1135
		// "test7.txt" number of lines/images: 1028
		List<Instance> test1 = Utilities.readDataSet(Constants.TEST1_PATH, Constants.LABEL_ONE);
		List<Instance> test7 = Utilities.readDataSet(Constants.TEST7_PATH, Constants.LABEL_SEVEN);
		List<Instance> tests = new ArrayList<>();
		tests.addAll(test1);
		tests.addAll(test7);

		/* Classify the Test data */
		double[] p = new double[tests.size()];

		// Test data
		for (int i = 0; i < tests.size(); i++) {
			// vector containing the data of each row
			int[] xtesti = tests.get(i).getX();
			p[i] = Utilities.classify(xtesti, weights);
		}

		// from xtest0 to xtest1134 -> ONE is the correct category
		// from xtest1135 to xtest2162 -> SEVEN is the correct category

		int wrong_counter = 0;
		for (int i = 0; i < tests.size(); i++) {
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

	// the gradient ascent algorithm
	public static double[] train(List<Instance> instances) {

		/** the weights to train */
		double[] weights = new double[N + 1];

		// the first weight is set to 1
		weights[0] = 1;

		double lik_old = Integer.MIN_VALUE;
		for (int iter = 0; iter < maxiters; iter++) {
			double lik = 0.0;
			for (int i = 0; i < M; i++) { // for all train examples
				int[] x = instances.get(i).getX();
				double predicted = Utilities.classify(x, weights);
				int label = instances.get(i).getLabel();
				for (int j = 0; j < N + 1; j++) { // for all features
					double grad = (label - predicted) * x[j] / M;
					// update all the weights simultaneously
					weights[j] = weights[j] + alpha * grad;
				}

				// Compute the likelihood estimate.
				// We want to maximize this.
				lik += label * Math.log(predicted) + (1 - label) * Math.log(1 - predicted);
			}

			System.out.println("iteration: " + (iter + 1) + ", likelihood estimate: " + lik);

			// Check for convergence.
			if (Math.abs(lik - lik_old) < tol) {
				break;
			}
			lik_old = lik;
		}

		return weights;
	}


}
