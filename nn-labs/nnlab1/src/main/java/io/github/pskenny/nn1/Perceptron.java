package io.github.pskenny.nn1;

import java.util.*;

public class Perceptron {
	private float[] weights;
	private float theta = 0.2f;

	public Perceptron(int connection) {
		weights = new float[connection];

		// Generate random initial weights
		for (int i = 0; i < weights.length; i++) {
			// (((number between 0 and 1) * 2) - 1) = number between -1 and 1
			weights[i] = random.nextFloat() * 2 - 1;
		}

		System.out.println(this);
	}

	public float[] getWeights() {
		return weights;
	}

	public int activate(float[] inputs) {
		/*
		 * Apply weights to inputs. Assumes number of inputs does not excede number of
		 * weights. Weights range from -1 to 1. This produces a range of outputs, from
		 * the original input to it's negation, including everything in-between.
		 */
		float sum = 0;
		for (int i = 0; i < weights.length; i++) {
			sum += inputs[i] * weights[i];
		}

		/*
		 * Perform a hard limit. Subtract threshold (theta) from sum. If it's above 0
		 * say it's a 1, if not, say it's 0. This function produces a step graph.
		 */
		return sum - theta >= 0 ? 1 : 0;
	}

	public String toString() {
		return "Perceptron [weights=" + Arrays.toString(weights) + "]";
	}

	// Trainer...
	private float alpha = 0.1f; // Learning rate
	private Random random = new Random();

	public void train(float[][] inputs, float[] expected, int max_epochs) {
		/*
		 * Note: Why is inputs two dimensional? Each singular activation function
		 * operation of this perceptron takes in any number of inputs and gives a
		 * singular output. Training data specifies 2 inputs. This example is tailored
		 * for modeling AND and OR operations which makes this perceptron an analogue
		 * for an AND or OR gates, depending on training data given.
		 */
		boolean hasError = true;
		float error = 1000;
		int epoch = 0;

		// Loop as long as epochs (max itertions) isn't reached and errors occur
		while (epoch < max_epochs && hasError) {
			// Reset error flag
			hasError = false;

			for (int row = 0; row < inputs.length; row++) {
				int actual = activate(inputs[row]);
				error = expected[row] - actual;
				if (error != 0)
					hasError = true;

				// Adjust weights based on weightChange * input
				// alpha * error weights the change to make based on how high the error is
				for (int i = 0; i < weights.length; i++) {
					weights[i] += alpha * error * inputs[row][i];
				}
			}
			epoch++;
		}

		System.out.println("Training complete in " + epoch + " epochs.");
		System.out.println(this);
	}
}