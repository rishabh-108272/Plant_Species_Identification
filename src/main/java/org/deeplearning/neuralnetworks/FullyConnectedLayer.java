package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Random;

public class FullyConnectedLayer {
    private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
    private INDArray weights; // Weight matrix
    private INDArray biases; // Bias vector
    private INDArray outputs; // Outputs after activation
//    private INDArray input; // Inputs to the layer
//    private INDArray deltas; // Deltas for backpropagation
    public double learningRate;
    private ActivationFunction activationFunction;
    private int numNeurons;

    public FullyConnectedLayer( int numNeurons, ActivationFunction activationFunction, double learningRate) {
        this.activationFunction = activationFunction;
        this.learningRate=learningRate;
        this.numNeurons = numNeurons;

//        // Xavier initialization
//        double limit = Math.sqrt(6.0 / (numInputs + numNeurons));
//        weights = Nd4j.rand(numNeurons, numInputs).muli(2 * limit).subi(limit); // Shape: [numNeurons, numInputs]
//
//// If necessary, reshape the weights
//        weights = weights.reshape(numInputs, numNeurons); // Shape: [48387, 128]
//
//        biases = Nd4j.zeros(numNeurons, 1); // Shape: [128, 1] - Initialize biases to zero

    }

    public INDArray forward(INDArray input) {

        System.out.println("[DENSE FORWARD PASS]"+Arrays.toString(input.shape()));

        // Initialize Parameters
        if(this.biases == null)
            this.biases = Nd4j.rand(1,this.numNeurons);
        if(this.weights == null){
            double limit = Math.sqrt(6.0 / input.size(1) + numNeurons);
            long[] WeightShape = new long[]{input.size(1), (long) this.numNeurons};
            weights = Nd4j.rand(WeightShape).muli(2*limit).subi(limit);
        }

//        this.input=input;
        // Flatten the 4D MaxPooling output to 2D [batchSize, channels * pooledHeight * pooledWidth]
//        INDArray flattenedInput = input.reshape(input.size(0), input.length() / input.size(0));
//        System.out.println("Flattened input shape for FC layer: " + flattenedInput.shapeInfoToString());

        // Ensure weights are compatible with the flattened input
//        if (weights.size(0) != flattenedInput.size(1)) {
//            throw new IllegalStateException("Weights matrix has an incompatible shape for the input. " +
//                    "Expected weights rows: " + flattenedInput.size(1) + ", but got: " + weights.size(0));
//        }

        // Proceed with forward pass
        INDArray weightedSum = input.mmul(weights).add(this.biases);

        // Ensure biases has the shape [1, numNeurons] to be compatible with weightedSum
//        if (biases.shape()[0] != 1 || biases.shape()[1] != weights.columns()) {
//            biases = biases.reshape(1, weights.columns());
//        }

        // Add biases to each row of weightedSum
//        weightedSum.addiRowVector(biases);
        this.outputs = activationFunction.activate(weightedSum);

        System.out.println("[DENSE FORWARD PASS COMPLETED]"+Arrays.toString(outputs.shape()));
        return outputs;
    }



    public INDArray backward(INDArray dL, INDArray InputActivations) {

        System.out.println("[DENSE BACKWARD PASS]"+Arrays.toString(dL.shape()));

        INDArray deltas = dL.mul(activationFunction.derivative(outputs));
//        deltas = dL.mmul(getWeights().dup().transpose()).mul(activationFunction.derivative(outputs));
//        INDArray weightedSumDerivative = activationFunction.derivative(outputs);

        // Reshape weightedSumDerivative for element-wise multiplication compatibility
//        weightedSumDerivative = weightedSumDerivative.reshape(1, weightedSumDerivative.length()).transpose();
//        System.out.println("Shape of weightedSumDerivative: " + weightedSumDerivative.shapeInfoToString());

        // Print current shapes for debugging
//        System.out.println("Shape of nextLayerDeltas: " + nextLayerDeltas.shapeInfoToString());
//        System.out.println("Shape of nextLayerWeights: " + nextLayerWeights.shapeInfoToString());

        // Calculate the expected number of rows in nextLayerDeltas for matrix multiplication
//        long nextLayerWeightsColumns = nextLayerWeights.columns();  // This is 128 based on your previous info

        // Check if nextLayerDeltas can be reshaped into the correct number of rows
//        if (nextLayerDeltas.rows() != nextLayerWeightsColumns) {
//            // Reshape nextLayerDeltas to be [128, 1] for compatibility with nextLayerWeights transpose
//            nextLayerDeltas = nextLayerDeltas.reshape(nextLayerWeightsColumns, 1);
////            System.out.println("Reshaped nextLayerDeltas to: " + nextLayerDeltas.shapeInfoToString());
//        }

        // Compute deltas for this layer
//        deltas = nextLayerWeights.transpose().mmul(nextLayerDeltas).mul(weightedSumDerivative);

        // Update Parameters
        INDArray dWeights = InputActivations.transpose().mmul(deltas).div(InputActivations.size(0));
        INDArray dBiases = deltas.sum(0).div(InputActivations.size(0));

        weights.subi(dWeights.mul(learningRate));
        biases.subi(dBiases.mul(learningRate));

        deltas = deltas.mmul(getWeights().transpose());
        System.out.println("[DENSE BACKWARD PASS COMPLETED]"+Arrays.toString(deltas.shape()));
        return deltas;
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

}
