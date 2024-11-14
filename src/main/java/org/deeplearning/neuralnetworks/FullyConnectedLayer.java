package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class FullyConnectedLayer {
    private INDArray weights; // Weight matrix
    private INDArray biases; // Bias vector
    private INDArray outputs; // Outputs after activation
    private INDArray input; // Inputs to the layer
    private INDArray deltas; // Deltas for backpropagation
    public double learningRate;
    private ActivationFunction activationFunction;

    public FullyConnectedLayer(int numInputs, int numNeurons, ActivationFunction activationFunction, double learningRate) {
        this.activationFunction = activationFunction;
        this.learningRate=learningRate;
        // Xavier initialization
        double limit = Math.sqrt(6.0 / (numInputs + numNeurons));
        weights = Nd4j.rand(numNeurons, numInputs).muli(2 * limit).subi(limit); // Shape: [numNeurons, numInputs]

// If necessary, reshape the weights
        weights = weights.reshape(numInputs, numNeurons); // Shape: [48387, 128]

        biases = Nd4j.zeros(numNeurons, 1); // Shape: [128, 1] - Initialize biases to zero

    }

    public INDArray forward(INDArray input) {
        System.out.println("Fully Connected Layer input shape (before flattening): " + input.shapeInfoToString());
        this.input=input;
        // Flatten the 4D MaxPooling output to 2D [batchSize, channels * pooledHeight * pooledWidth]
        INDArray flattenedInput = input.reshape(input.size(0), input.length() / input.size(0));
        System.out.println("Flattened input shape for FC layer: " + flattenedInput.shapeInfoToString());

        // Ensure weights are compatible with the flattened input
        if (weights.size(0) != flattenedInput.size(1)) {
            throw new IllegalStateException("Weights matrix has an incompatible shape for the input. " +
                    "Expected weights rows: " + flattenedInput.size(1) + ", but got: " + weights.size(0));
        }

        // Proceed with forward pass
        INDArray weightedSum = flattenedInput.mmul(weights);

        // Ensure biases has the shape [1, numNeurons] to be compatible with weightedSum
        if (biases.shape()[0] != 1 || biases.shape()[1] != weights.columns()) {
            biases = biases.reshape(1, weights.columns());
        }

        // Add biases to each row of weightedSum
        weightedSum.addiRowVector(biases);
        this.outputs = activationFunction.activate(weightedSum);

        System.out.println("Output shape for FC layer: " + outputs.shapeInfoToString());
        System.out.println("FC layer forward pass completed!");
        return outputs;
    }



    public INDArray backward(INDArray nextLayerDeltas, INDArray nextLayerWeights) {
        INDArray weightedSumDerivative = activationFunction.derivative(outputs);

        // Reshape weightedSumDerivative for element-wise multiplication compatibility
        weightedSumDerivative = weightedSumDerivative.reshape(1, weightedSumDerivative.length()).transpose();
        System.out.println("Shape of weightedSumDerivative: " + weightedSumDerivative.shapeInfoToString());

        // Print current shapes for debugging
        System.out.println("Shape of nextLayerDeltas: " + nextLayerDeltas.shapeInfoToString());
        System.out.println("Shape of nextLayerWeights: " + nextLayerWeights.shapeInfoToString());

        // Calculate the expected number of rows in nextLayerDeltas for matrix multiplication
        long nextLayerWeightsColumns = nextLayerWeights.columns();  // This is 128 based on your previous info

        // Check if nextLayerDeltas can be reshaped into the correct number of rows
        if (nextLayerDeltas.rows() != nextLayerWeightsColumns) {
            // Reshape nextLayerDeltas to be [128, 1] for compatibility with nextLayerWeights transpose
            nextLayerDeltas = nextLayerDeltas.reshape(nextLayerWeightsColumns, 1);
            System.out.println("Reshaped nextLayerDeltas to: " + nextLayerDeltas.shapeInfoToString());
        }

        // Compute deltas for this layer
        deltas = nextLayerWeights.transpose().mmul(nextLayerDeltas).mul(weightedSumDerivative);

        // Debug output for deltas shape
        System.out.println("Computed deltas shape: " + deltas.shapeInfoToString());

        // Update weights and biases
        System.out.println("Updating weights....");
        updateWeights(learningRate);

        System.out.println("FC layer Backward pass completed!!!");
        return deltas;
    }




    // Update weights and biases
    public void updateWeights(double learningRate) {
        // Compute gradients with respect to weights and biases
        INDArray weightGradients = deltas.mmul(input.transpose());  // Assuming 'input' is stored from forward pass
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
        return input;
    }

    public INDArray getDeltas() {
        return deltas;
    }
}
