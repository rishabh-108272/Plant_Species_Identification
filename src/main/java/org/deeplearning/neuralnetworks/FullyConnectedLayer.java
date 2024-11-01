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
        weights = Nd4j.rand(numNeurons, numInputs).muli(2 * limit).subi(limit); // Uniform distribution between [-limit, limit]
        biases = Nd4j.zeros(numNeurons, 1); // Initialize biases to zero
    }

    // Forward pass for the fully connected layer
    public INDArray forward(INDArray input) {
        this.inputs = input;
        INDArray weightedSum = weights.mmul(input).addColumnVector(biases);
        this.outputs = activationFunction.activate(weightedSum);
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
