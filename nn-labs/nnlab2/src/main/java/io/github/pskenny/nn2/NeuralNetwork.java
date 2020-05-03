package io.github.pskenny.nn2;

/*
 * ------------------------------------------------------------------------
 * B.Sc. (Hons) in Software Development - Artificial Intelligence
 * ------------------------------------------------------------------------
 * 
 * This 3-lay neural network is nothing more than a set of three arrays to 
 * represent the values of the input, hidden and output layers. This design 
 * could easily be extended to allow an n-layer neural network to be created. 
 * Moreover, the speed of the feed-forward and back-propagation operations 
 * can be significantly increased by exploiting matrix multiplication.  
 */

import java.util.*;
import io.github.pskenny.nn2.activator.*;

public class NeuralNetwork {
	private Activator activator;
	private double[] inputs; // Stores inputs X1, X2,...,Xn
	private double[] hiddenInputs; // Stores activated inputs
	private double[] outputs; // Stores Y

	private double[][] inputToHiddenLayerWeights; // Matrix of weights for input->hidden layer
	private double[][] hiddenToOutputLayerWeights; // Matrix of weights for hidden->output layer

	public NeuralNetwork(Activator.ActivationFunction function, int numInputNodes, int numHiddenNodes,
			int numOutputNodes) {
		this.activator = ActivatorFactory.getInstance().getActivator(function);
		this.inputs = new double[numInputNodes];
		this.hiddenInputs = new double[numHiddenNodes];
		this.outputs = new double[numOutputNodes];

		this.inputToHiddenLayerWeights = new double[numInputNodes + 1][numHiddenNodes]; // An extra row for the bias
		this.hiddenToOutputLayerWeights = new double[numHiddenNodes + 1][numOutputNodes]; // An extra row for the bias

		this.initialiseWeights(inputToHiddenLayerWeights);
		this.initialiseWeights(hiddenToOutputLayerWeights);
	}

	public double[] process(double[] dataInputs) throws Exception {
		// Check for consistent input
		if (dataInputs.length != inputs.length) {
			throw new Exception("Invalid input for a " + inputs.length + "x" + hiddenInputs.length + "x"
					+ outputs.length + " neural network.");
		}

		// Initialise the input layer
		for (int i = 0; i < inputs.length; i++) {
			inputs[i] = dataInputs[i];
		}

		// Feed the inputs forward to the next layer
		this.feedForward();

		// Return the out layer
		return outputs;
	}

	private void initialiseWeights(double[][] matrix) {
		// Initialise weights to random numbers in range -0.5 - +0.5
		Random rand = new Random();
		for (int row = 0; row < matrix.length; ++row) {
			for (int col = 0; col < matrix[row].length; ++col) {
				matrix[row][col] = rand.nextDouble() - 0.5;
			}
		}
	}

	public void feedForward() {
		computeInputToHiddenLayer();
		computeHiddenToOutputLayer();
	}

	private void computeInputToHiddenLayer() {
		// Feed the inputs forward through the network as a weighted sum
		double sum = 0.0d;
		// Compute Input->Hidden Layer
		for (int hiddenIndex = 0; hiddenIndex < hiddenInputs.length; ++hiddenIndex) {
			sum = 0.0d;
			for (int inputIndex = 0; inputIndex < inputs.length; ++inputIndex)
				sum += inputs[inputIndex] * inputToHiddenLayerWeights[inputIndex][hiddenIndex];
				
			sum += inputToHiddenLayerWeights[inputs.length][hiddenIndex];
			hiddenInputs[hiddenIndex] = activator.activate(sum); // Apply activation function
		}
	}

	private void computeHiddenToOutputLayer() {
		// Feed the inputs forward through the network as a weighted sum
		double sum = 0.0d;
		// Compute Hidden->Output Layer
		for (int outputIndex = 0; outputIndex < outputs.length; ++outputIndex) {
			sum = 0.0d;
			for (int hiddenIndex = 0; hiddenIndex < hiddenInputs.length; ++hiddenIndex)
				sum += hiddenInputs[hiddenIndex] * hiddenToOutputLayerWeights[hiddenIndex][outputIndex];

			sum += hiddenToOutputLayerWeights[hiddenInputs.length][outputIndex];
			outputs[outputIndex] = activator.activate(sum); // Apply activation function
		}
	}

	public Activator getActivator() {
		return activator;
	}

	public double[] getInputLayer() {
		return inputs;
	}

	public void setInputs(double[] inputs) {
		this.inputs = inputs;
	}

	public double[] getHiddenLayer() {
		return hiddenInputs;
	}

	public double[] getOutputLayer() {
		return outputs;
	}

	public double[][] getHiddenWeights() {
		return inputToHiddenLayerWeights;
	}

	public double[][] getOutputWeights() {
		return hiddenToOutputLayerWeights;
	}
}