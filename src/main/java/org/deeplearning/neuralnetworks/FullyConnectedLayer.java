package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class FullyConnectedLayer {
    private INDArray weights; // Weight matrix
    private INDArray biases; // Bias vector
    private INDArray outputs; // Outputs after activation
    private INDArray inputs; // Inputs to the layer
    private INDArray deltas; // Deltas for backpropagation
    private ActivationFunction activationFunction;

    public FullyConnectedLayer(int numInputs, int numNeurons, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        Random random = new Random();

        // Xavier initialization
        double limit = Math.sqrt(6.0 / (numInputs + numNeurons));
        weights = Nd4j.rand(numNeurons, numInputs).muli(2 * limit).subi(limit); // Shape: [numNeurons, numInputs]

// If necessary, reshape the weights
        weights = weights.reshape(numInputs, numNeurons); // Shape: [131072, 128]

        biases = Nd4j.zeros(numNeurons, 1); // Shape: [128, 1] - Initialize biases to zero

    }

    // Forward pass for the fully connected layer
    public INDArray forward(INDArray input) {
        System.out.println("Weights shape:"+ weights.shapeInfoToString());

// Transpose the input if necessary to align dimensions
        input = input.transpose();
        System.out.println("Input shape after reshaping for fully connected layer: " + input.shapeInfoToString());

// Proceed with forward pass
        INDArray weightedSum = weights.transpose().mmul(input).addColumnVector(biases);
        this.outputs = activationFunction.activate(weightedSum.transpose());// Transpose back if needed
        System.out.println("Output shape for FC layer:"+outputs.shapeInfoToString());
        System.out.println("FC layer Forward pass completed!!!");
        return outputs;
    }

    // Backward pass for the fully connected layer
    public INDArray backward(INDArray nextLayerDeltas, INDArray nextLayerWeights) {
        INDArray weightedSumDerivative = activationFunction.derivative(outputs);
        deltas = nextLayerWeights.transpose().mmul(nextLayerDeltas).mul(weightedSumDerivative);
        return deltas;
    }

    // Update weights and biases
    public void updateWeights(double learningRate) {
        INDArray weightGradients = deltas.mmul(inputs.transpose());
        weights.subi(weightGradients.mul(learningRate));
        biases.subi(deltas.mul(learningRate));
    }

    public INDArray getWeights() {
        return weights;
    }

    public INDArray getBiases() {
        return biases;
    }

    public INDArray getOutputs() {
        return outputs;
    }

    public INDArray getInputs() {
        return inputs;
    }

    public INDArray getDeltas() {
        return deltas;
    }
}
