package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Utilities {

	// N: the number features
	private static final int N = Constants.NUM_PIXELS;

	public static double classify(int[] x, double[] weights) {
		double logit = 0.0;
		for (int i = 0; i < N + 1; i++) {
			logit += x[i] * weights[i];
		}
		return sigmoid(logit);
	}

	// sigmoid function
	public static double sigmoid(double x) {
		return (1 / (1 + Math.exp(-x)));
	}


	// returns the sum of all elements of the given array
	public static double sumOfSquares(double[] x) {
		double sum = 0;
		// we start from index 1
		for (int i = 1; i < x.length; i++) {
			sum += Math.pow(x[i], 2);
		}
		return sum;
	}

	// for reading Train & Test data
	public static List<Instance> readDataSet(String file, int label) {
		List<Instance> data = new ArrayList<>();

		try {
			BufferedReader br = new BufferedReader(new FileReader(file));

			String line;

			while ((line = br.readLine()) != null) {
				String[] stringline = line.split(" ");
				int N = stringline.length;
				int[] intline = new int[N + 1];

				// the first feature is always 1
				intline[0] = 1;
				for (int i = 0; i < N; i++) {
					intline[i + 1] = Integer.parseInt(stringline[i]);
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
