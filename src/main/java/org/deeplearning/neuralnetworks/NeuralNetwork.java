package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class NeuralNetwork {
    private ArrayList<Layer> layers;
    private double learningRate;

    public NeuralNetwork(int[] layersSizes, double learningRate, ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction) {
        layers = new ArrayList<>();
        this.learningRate = learningRate;

        for (int i = 0; i < layersSizes.length; i++) {
            int numInputs = (i == 0) ? layersSizes[i] : layersSizes[i - 1];
            // Use the hidden activation function for all layers except the last one
            ActivationFunction activationFunction = (i == layersSizes.length - 1) ? outputActivationFunction : hiddenActivationFunction;
            layers.add(new Layer(layersSizes[i], numInputs, activationFunction));
        }
    }

    public INDArray feedForward(INDArray inputs) {
        INDArray outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.getOutputs(outputs);
        }
        return outputs;
    }

    public void backpropagate(INDArray inputs, INDArray expectedOutputs) {
        INDArray actualOutputs = feedForward(inputs);
        INDArray errors = expectedOutputs.sub(actualOutputs); // This should be of shape (1, 1) if using one output neuron.

        // Backpropagate errors
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            INDArray nextErrors = Nd4j.zeros(layer.getNeurons().size(), 1); // Correct size for nextErrors

            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                INDArray activationOutput = neuron.getOutput(); // Get the output as INDArray
                INDArray activationDerivative = neuron.getActivationFunction().derivative(activationOutput); // Use INDArray for derivative

                // Ensure correct indexing for errors
                if (j < errors.length()) { // Check that we're within bounds for errors
                    INDArray neuronDelta = Nd4j.scalar(errors.getDouble(j)).mul(activationDerivative); // Calculate delta as INDArray
                    neuron.setDelta(neuronDelta);
                } else {
                    // Log or handle the case where j is out of bounds
                    System.err.println("Warning: Index " + j + " is out of bounds for errors.");
                    continue; // Skip this iteration if there's an issue
                }

                // Update nextErrors based on the neuron's weights and delta
                for (int k = 0; k < neuron.getWeights().rows(); k++) {
                    nextErrors.putScalar(k, nextErrors.getDouble(k) + neuron.getWeights().getDouble(k) * neuron.getDelta().getDouble(0)); // Use getDouble(0) to get scalar value
                }
            }

            if (i > 0) {
                errors = nextErrors; // Propagate errors to previous layer
            }
        }

        // Adjust weights and biases
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            INDArray layerInputs = (i == 0) ? inputs : layers.get(i - 1).getOutputs(inputs);

            for (Neuron neuron : layer.getNeurons()) {
                neuron.adjustWeights(learningRate, layerInputs);
            }
        }
    }


    public void train(double[][] inputs, double[][] outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.print("Epoch:" + epoch);
            for (int i = 0; i < inputs.length; i++) {
                INDArray input = Nd4j.create(inputs[i]);
                INDArray output = Nd4j.create(outputs[i]);
                backpropagate(input, output);
                System.out.print("=");
            }
            System.out.println();
        }
    }

    public void test(double[][] inputs, double[][] outputs, int epochs) {
        int correctCount = 0;
        for (int i = 0; i < inputs.length; i++) {
            INDArray input = Nd4j.create(inputs[i]);
            INDArray expectedOutput = Nd4j.create(outputs[i]);
            INDArray predictedOutput = feedForward(input);

            if (isOutputCorrect(predictedOutput, expectedOutput)) {
                correctCount++;
            }
        }

        double accuracy = (double) correctCount / inputs.length * 100;
        System.out.printf("Accuracy over %d epochs: %.2f%%%n", epochs, accuracy);
    }

    private boolean isOutputCorrect(INDArray predictedOutputs, INDArray expectedOutputs) {
        int predicted = (predictedOutputs.getDouble(0) > 0.5) ? 1 : 0;
        int expected = (expectedOutputs.getDouble(0) > 0.5) ? 1 : 0;
        return predicted == expected;
    }

    public double[][] predict(double[][] inputs) {
        double[][] predictions = new double[inputs.length][];

        for (int i = 0; i < inputs.length; i++) {
            INDArray input = Nd4j.create(inputs[i]);
            INDArray output = feedForward(input);
            predictions[i] = output.toDoubleVector(); // Convert INDArray to double array
        }

        return predictions;
    }
}
