package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;

public class ConvolutionalLayer {
    private INDArray[] filters; // Array of filters
    private int numFilters; // Number of filters
    private int filterSize; // Size of each filter (assumed to be square)
    private int stride; // Stride for convolution
    private int padding; // Padding for convolution
    private ActivationFunction activationFunction;

    public ConvolutionalLayer(int numFilters, int filterSize, int stride, int padding, ActivationFunction activationFunction) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.padding = padding;
        this.activationFunction = activationFunction;
        this.filters = new INDArray[numFilters];

        // Initialize filters with random values
        Random random = new Random();
        for (int i = 0; i < numFilters; i++) {
            filters[i] = Nd4j.rand(filterSize, filterSize).subi(0.5); // Initialize filters with values between -0.5 and 0.5
        }
    }

    private INDArray applyFilter(INDArray input, INDArray filter) {
        // Extract the dimensions from the 4D tensor: [batch_size, channels, height, width]
        long batchSize = input.size(0);
        long channels = input.size(1);
        long inputHeight = input.size(2);
        long inputWidth = input.size(3);
        long outputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
        long outputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;

        // Create an output tensor with the same batch size and number of channels
        INDArray output = Nd4j.create(batchSize, channels, outputHeight, outputWidth);

        // Iterate over each image in the batch
        for (int b = 0; b < batchSize; b++) {
            // Iterate over each channel
            for (int c = 0; c < channels; c++) {
                INDArray inputChannel = input.get(NDArrayIndex.point(b), NDArrayIndex.point(c));
                INDArray paddedInput = padInput(inputChannel, padding);

                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        // Extract the sub-matrix for the current position
                        int startX = i * stride;
                        int startY = j * stride;
                        INDArray subMatrix = paddedInput.get(NDArrayIndex.interval(startX, startX + filterSize), NDArrayIndex.interval(startY, startY + filterSize));

                        // Perform element-wise multiplication and sum the result
                        double convolvedValue = subMatrix.mul(filter).sumNumber().doubleValue();

                        // Apply activation function
                        output.putScalar(new int[]{b, c, i, j}, activationFunction.activate(Nd4j.scalar(convolvedValue)).getDouble(0));
                    }
                }
            }
        }
        return output;
    }


    // Manually pad the input
    private INDArray padInput(INDArray input, int padding) {
        if (padding == 0) {
            return input;
        }

        int inputHeight = input.rows();
        int inputWidth = input.columns();
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;

        INDArray paddedInput = Nd4j.zeros(paddedHeight, paddedWidth);
        paddedInput.put(new INDArrayIndex[]{
                NDArrayIndex.interval(padding, padding + inputHeight),
                NDArrayIndex.interval(padding, padding + inputWidth)
        }, input);
        return paddedInput;
    }

    // Forward pass for the convolutional layer
    public INDArray[] forward(INDArray input) {
        INDArray[] featureMaps = new INDArray[numFilters];

        for (int i = 0; i < numFilters; i++) {
            featureMaps[i] = applyFilter(input, filters[i]);
        }
        return featureMaps;
    }

    // Backward pass for the convolutional layer
    public INDArray backward(INDArray error, double learningRate) {
        // Create an array to hold gradients for each filter
        INDArray[] filterGradients = new INDArray[numFilters];

        // Initialize input gradients
        INDArray inputGradients = Nd4j.zeros(error.rows(), error.columns());

        // Calculate gradients for each filter
        for (int i = 0; i < numFilters; i++) {
            filterGradients[i] = calculateFilterGradient(error, i); // Calculate the gradient for the i-th filter
        }

        // Update weights based on gradients
        for (int i = 0; i < numFilters; i++) {
            filters[i].subi(filterGradients[i].mul(learningRate)); // Update filter weights using the learning rate
        }

        // Calculate input gradients based on the error and the filters
        for (int i = 0; i < numFilters; i++) {
            // Use the filter gradients to compute the input gradients
            for (int j = 0; j < error.rows(); j++) {
                for (int k = 0; k < error.columns(); k++) {
                    // Calculate start position for input gradient accumulation
                    int startX = j * stride;
                    int startY = k * stride;
                    inputGradients.get(NDArrayIndex.interval(startX, startX + filterSize), NDArrayIndex.interval(startY, startY + filterSize))
                            .addi(filterGradients[i].mul(error.getDouble(j, k))); // Accumulate input gradients
                }
            }
        }

        return inputGradients; // Return the input gradients for the next layer
    }

    private INDArray calculateFilterGradient(INDArray error, int filterIndex) {
        int inputHeight = error.rows();
        int inputWidth = error.columns();
        INDArray filterGradient = Nd4j.zeros(filters[filterIndex].shape());

        for (int j = 0; j < inputHeight; j++) {
            for (int k = 0; k < inputWidth; k++) {
                // Calculate the start position for the corresponding area in the input
                int startX = j * stride;
                int startY = k * stride;

                // Extract the relevant area from the input
                INDArray subMatrix = error.get(NDArrayIndex.point(j), NDArrayIndex.point(k)); // Error at position (j,k)
                INDArray inputPatch = Nd4j.zeros(filterSize, filterSize);
                inputPatch = inputPatch.add(padInput(subMatrix, padding)); // Get the corresponding input area (make sure to pad as needed)

                // Accumulate the contributions to the filter gradient
                filterGradient.addi(inputPatch.mul(subMatrix)); // Element-wise multiplication to get the gradient
            }
        }

        return filterGradient; // Return the computed gradient for the filter
    }


    public INDArray[] getFilters() {
        return filters;
    }

    public int getNumFilters() {
        return numFilters;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }
}
