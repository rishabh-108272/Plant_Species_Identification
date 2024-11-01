package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Random;

public class Neuron {
    private INDArray weights;
    private double bias;
    private INDArray output; // Changed to INDArray for consistency
    private INDArray delta;  // Changed to INDArray for consistency
    private ActivationFunction activationFunction;

    public Neuron(int numInputs, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        Random random = new Random();

        // Xavier initialization
        double limit = Math.sqrt(6.0 / (numInputs + 1)); // numInputs + 1 for bias

        // Initialize weights with DOUBLE data type
        weights = Nd4j.rand(DataType.DOUBLE, numInputs, 1).muli(2 * limit).subi(limit);

        // Initialize bias to a small random value
        bias = 0.01;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public INDArray getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public INDArray getOutput() {
        return output; // Return output as INDArray
    }

    public void setOutput(INDArray output) {
        this.output = output; // Accept output as INDArray
    }

    public INDArray getDelta() {
        return delta; // Return delta as INDArray
    }

    public void setDelta(INDArray delta) {
        this.delta = delta; // Accept delta as INDArray
    }

    public INDArray activate(INDArray inputs) {
        // Ensure inputs are a column vector and have DOUBLE data type
        INDArray reshapedInputs = inputs.reshape(inputs.length(), 1).castTo(DataType.DOUBLE);
        INDArray activation = weights.transpose().mmul(reshapedInputs).add(bias); // Changed to INDArray operation
        output = activationFunction.activate(activation); // Keep activation as INDArray
        return output;
    }

    public void calculateDelta(INDArray error) {
        // Update delta using the derivative of the activation function and the error
        this.delta = error.mul(activationFunction.derivative(output)); // Assume error is an INDArray
    }

    public void adjustWeights(double learningRate, INDArray inputs) {
        // Ensure inputs are in DOUBLE data type
        INDArray reshapedInputs = inputs.reshape(inputs.length(), 1).castTo(DataType.DOUBLE);
        weights.addi(reshapedInputs.mul(delta).muli(learningRate)); // Use INDArray for weight adjustment
        bias += learningRate * delta.getDouble(0); // Update bias using scalar value from delta
    }
}
