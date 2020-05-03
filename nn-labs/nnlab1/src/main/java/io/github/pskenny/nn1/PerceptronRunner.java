package io.github.pskenny.nn1;

public class PerceptronRunner {

	public PerceptronRunner() {
		// Note: data is already normalised
		float[][] data = { 
			{ 0.00f, 0.00f }, 
			{ 1.00f, 0.00f }, 
			{ 0.00f, 1.00f }, 
			{ 1.00f, 1.00f } 
		};
		float[] expectedAnd = {0.00f, 0.00f, 0.00f, 1.00f}; // Logical AND
		float[] expectedOr = { 0.00f, 1.00f, 1.00f, 1.00f }; // Logical OR
		int epoch = 10000;

		// Note: It is bad practise to use training data as testing data. Don't do this.
		System.out.println("Train for AND");
		trainAnd(data, expectedAnd, epoch);
		System.out.println("\nTrain for OR");
		trainOr(data, expectedOr, epoch);
	}

	/**
	 * Train perceptron as AND function. Assumes same number of inputs and expected outputs.
	 * 
	 * @param data Inputs for training and testing.
	 * @param expected Expected outputs during training.
	 * @param epoch Number of iterations during training.
	 */ 
	private void trainAnd(float[][] data, float[] expected, int epoch) {
		Perceptron p = new Perceptron(2);
		p.train(data, expected, epoch);

		// Test
		for (int row = 0; row < data.length; ++row) {
			int result = p.activate(data[row]);
			System.out.println("Result " + row + ": " + result);
		}
	}

	/**
	 * Train perceptron as OR function. Assumes same number of inputs and expected outputs.
	 * 
	 * @param data Inputs for training and testing.
	 * @param expected Expected outputs during training.
	 * @param epoch Number of iterations during training.
	 */ 
	private void trainOr(float[][] data, float[] expected, int epoch) {
		Perceptron p = new Perceptron(2);
		p.train(data, expected, epoch);

		// Test
		for (int row = 0; row < data.length; ++row) {
			int result = p.activate(data[row]);
			System.out.println("Result " + row + ": " + result);
		}
	}

	public static void main(String[] args) {
		new PerceptronRunner();
	}
}