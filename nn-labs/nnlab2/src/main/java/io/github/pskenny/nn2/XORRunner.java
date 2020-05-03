package io.github.pskenny.nn2;

import io.github.pskenny.nn2.activator.*;

public class XORRunner {
    double[][] data = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    double[][] expected = { { 0 }, { 1 }, { 1 }, { 0 } };
    double[][] tests = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 } };

    public XORRunner() {
        NeuralNetwork nn = new NeuralNetwork(Activator.ActivationFunction.Sigmoid, 2, 2, 1);
        BackpropagationTrainer trainer = new BackpropagationTrainer(nn);

        // Perform training
        trainer.train(data, expected, 0.01, 1000000);

        tests(nn);
    }

    private void tests(NeuralNetwork nn) {
        for (double[] test : tests) {
            test(test, nn);
        }
    }

    private void test(double[] test, NeuralNetwork nn) {
        String testValues = Utils.doubleArrayToString(test);

        try {
            System.out.println(testValues + "=>" + getRoundedValue(nn.process(test)));
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    public static long getRoundedValue(double[] vector) {
        return Math.round(vector[0]);
    }

    public static void main(String[] args) {
        new XORRunner();
    }
}