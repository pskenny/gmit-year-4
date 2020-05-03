package io.github.pskenny.nn2;

/*
 * ------------------------------------------------------------------------
 * B.Sc. (Hons) in Software Development - Artificial Intelligence
 * ------------------------------------------------------------------------
 * 
 * A simple implementation if the back-propagation training algorithm. This
 * class is designed to work with a 3-layer neural network. 
 */
import java.text.DecimalFormat;

public class BackpropagationTrainer implements Trainator {
	private static final double MOMENTUM = 0.95; // Controls the rate of descent
	private NeuralNetwork neuralNet;
	private double[] outputLayerErrors; // Error values in the output layer
	private double[] hiddenLayerErrors; // Error values in the hidden layer

	public BackpropagationTrainer(NeuralNetwork network) {
		this.neuralNet = network;
		outputLayerErrors = new double[neuralNet.getOutputLayer().length];
		hiddenLayerErrors = new double[neuralNet.getHiddenLayer().length];
	}

	public void train(double[][] trainingData, double[][] desired, double alpha, int epochLimit) {
		DecimalFormat decimalFormat = new DecimalFormat("#.#########"); // Formatter for nice output
		long startTime = System.currentTimeMillis(); // Start the clock
		double errorTolerance = 0.001d; // Stop training when we reach this error
		boolean hasError = true;
		int epoch = 0;

		// Train until epoch limit reached or within error tolerance
		for (; epoch < epochLimit && hasError; ++epoch) {

			for (int index = 0; index < trainingData.length; ++index) {
				double[] sample = trainingData[index];
				double[] expected = desired[index];

				for (int i = 0; i < neuralNet.getInputLayer().length; ++i)
					neuralNet.getInputLayer()[i] = sample[i];
				for (int i = 0; i < neuralNet.getOutputLayer().length; ++i)
					neuralNet.getOutputLayer()[i] = expected[i];

				neuralNet.feedForward();
				backPropagate(expected, alpha);
			}

			// Check if errors are within tolerance
			hasError = isOutsideErrorTolerance(errorTolerance);
		}
		System.out.println("[INFO] Training completed in " + ((System.currentTimeMillis() - startTime) / 1000) + " seconds.");
		System.out.println("[INFO] Epochs: " + epoch);
		System.out.println("[INFO] Sum of Squares Error: " + decimalFormat.format(computeSumOfErrorSquares()));
	}

	private boolean isOutsideErrorTolerance(double errorTolerance) {
		double sumOfSquaresError = computeSumOfErrorSquares();
		return (Math.abs(sumOfSquaresError) <= errorTolerance) ? false : true; // Bail out of training
	}

	private double computeSumOfErrorSquares() {
		double sumOfSquaresError = 0.0d;
		for (int i = 0; i < outputLayerErrors.length; ++i)
			sumOfSquaresError += Math.pow(outputLayerErrors[i], 2);
		return sumOfSquaresError;
	}

	// The comments use the same notation as used in the lecture notes
	private void backPropagate(double[] expected, double alpha) {
		computeOutputLayerErrorGradient(expected);
		computeHiddenLayerErrorGradient();
		// Alpha is an amount which error correcting (gradient descent) can be tweaked by. Bigger = bigger change and visa versa
		updateOutputLayerWeights(alpha);
		updateHiddenLayerWeights(alpha);
	}

	private void computeOutputLayerErrorGradient(double[] expected) {
		for (int i = 0; i < neuralNet.getOutputLayer().length; ++i)
			// delta_k(p) = y_k(p) x (1 - y_k(p)) * e_k(p)
			outputLayerErrors[i] = neuralNet.getActivator().derivative(neuralNet.getOutputLayer()[i])
					* (expected[i] - neuralNet.getOutputLayer()[i]);
	}

	private void computeHiddenLayerErrorGradient() {
		for (int hiddenIndex = 0; hiddenIndex < neuralNet.getHiddenLayer().length; ++hiddenIndex) {
			hiddenLayerErrors[hiddenIndex] = 0.0d;

			// delta_j(p) = y_j(p) * (1 - y_j(p)) * Sum(delta_k(p) * w_jk(p))
			for (int outputIndex = 0; outputIndex < neuralNet.getOutputLayer().length; ++outputIndex)
				hiddenLayerErrors[hiddenIndex] += outputLayerErrors[outputIndex]
						* neuralNet.getOutputWeights()[hiddenIndex][outputIndex];

			hiddenLayerErrors[hiddenIndex] *= neuralNet.getActivator()
					.derivative(neuralNet.getHiddenLayer()[hiddenIndex]);
		}
	}

	private void updateOutputLayerWeights(double alpha) {
		for (int outputIndex = 0; outputIndex < neuralNet.getOutputLayer().length; ++outputIndex) {
			for (int hiddenIndex = 0; hiddenIndex < neuralNet.getHiddenLayer().length; ++hiddenIndex)
				// delta_w_jk(p) = alpha * y_j(p) * delta_k(p)
				neuralNet.getOutputWeights()[hiddenIndex][outputIndex] += alpha
						* neuralNet.getHiddenLayer()[hiddenIndex] * outputLayerErrors[outputIndex];

			neuralNet.getOutputWeights()[neuralNet.getHiddenLayer().length][outputIndex] += (MOMENTUM * alpha
					* outputLayerErrors[outputIndex]);
		}
	}

	private void updateHiddenLayerWeights(double alpha) {
		for (int hiddenIndex = 0; hiddenIndex < neuralNet.getHiddenLayer().length; ++hiddenIndex) {
			for (int inputIndex = 0; inputIndex < neuralNet.getInputLayer().length; ++inputIndex)
				// delta_w_ij(p) = alpha * x_i(p) * delta_j(p)
				neuralNet.getHiddenWeights()[inputIndex][hiddenIndex] += (alpha * neuralNet.getInputLayer()[inputIndex]
						* hiddenLayerErrors[hiddenIndex]);

			// w_ij(p + 1) = w_ij(p) + delta_w_ij(p)
			neuralNet.getHiddenWeights()[neuralNet.getInputLayer().length][hiddenIndex] += (MOMENTUM * alpha
					* hiddenLayerErrors[hiddenIndex]);
		}
	}
}