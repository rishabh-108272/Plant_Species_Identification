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

        // Create Sobel X kernel for a 3-channel input
        INDArray sobelX = Nd4j.create(new float[][] {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
        }).reshape(1, 3, 3); // Initial single channel, 3x3 kernel

// Expand sobelX across 3 channels
        INDArray sobelX3Channel = Nd4j.concat(0, sobelX, sobelX, sobelX).reshape(3, 3, 3); // Shape: [3, 3, 3]

// Create Sobel Y kernel for a 3-channel input
        INDArray sobelY = Nd4j.create(new float[][] {
                {-1, -2, -1},
                { 0,  0,  0},
                { 1,  2,  1}
        }).reshape(1, 3, 3); // Initial single channel, 3x3 kernel

// Expand sobelY across 3 channels
        INDArray sobelY3Channel = Nd4j.concat(0, sobelY, sobelY, sobelY).reshape(3, 3, 3); // Shape: [3, 3, 3]

// Assuming numFilters is set to 2 for Sobel X and Sobel Y
        INDArray flattenedSobelX = sobelX3Channel.reshape(1, 27);
        INDArray flattenedSobelY = sobelY3Channel.reshape(1, 27);

        filters[0] = flattenedSobelX;
        filters[1] = flattenedSobelY;



    }

    // Forward pass for the convolutional layer
    public INDArray forward(INDArray input) {
        int batchSize = (int) input.size(0);
        int inputChannels = (int) input.size(1);
        int inputHeight = (int) input.size(2);
        int inputWidth = (int) input.size(3);
        int outputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;

        // Initialize output feature maps
        INDArray output = Nd4j.zeros(batchSize, numFilters, outputHeight, outputWidth);
        System.out.println(output.shapeInfoToString());
        // Pad the input
        INDArray paddedInput = padInput(input, padding);
        System.out.println("paddedInput Shape: "+paddedInput.shapeInfoToString());
        // Extract patches from the input
        INDArray patches = Nd4j.create(new int[]{batchSize, inputChannels, filterSize, filterSize, outputHeight, outputWidth});
        System.out.println("Patches Shape: "+patches.shapeInfoToString());
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                int startX = i * stride;
                int startY = j * stride;
                patches.put(new INDArrayIndex[]{
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.point(i),
                        NDArrayIndex.point(j)
                }, paddedInput.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(startX, startX + filterSize),
                        NDArrayIndex.interval(startY, startY + filterSize)));
            }
        }

        // Reshape patches to [batchSize * outputHeight * outputWidth, inputChannels * filterSize * filterSize]
        patches = patches.permute(0, 4, 5, 1, 2, 3).reshape(batchSize * outputHeight * outputWidth, inputChannels * filterSize * filterSize);
        System.out.println("Reshaped Patches Shape: " + patches.shapeInfoToString());

// Apply convolution filters
        for (int f = 0; f < numFilters; f++) {
            // Reshape the filter to [inputChannels * filterSize * filterSize, 1]
            INDArray filter = filters[f].reshape(inputChannels * filterSize * filterSize, 1);
            System.out.println("Reshaped Filter Shape: " + filter.shapeInfoToString());

            // Multiply the reshaped filter with the reshaped patches
            INDArray convolved = patches.mmul(filter);
            System.out.println("Convolved Shape: "+ convolved.shapeInfoToString());
            // Reshape the convolved output to [batchSize, outputHeight, outputWidth]
            convolved = convolved.reshape(batchSize, outputHeight, outputWidth);
            System.out.println("Convolved reshaped: "+ convolved.shapeInfoToString());
            convolved = activationFunction.activate(convolved);

            // Place the convolved result in the output array
            output.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(f), NDArrayIndex.all(), NDArrayIndex.all()}, convolved);
        }

        System.out.println("Shape of output: "+output.shapeInfoToString());
        return output;
    }

    // Manually pad the input
    private INDArray padInput(INDArray input, int padding) {
        if (padding == 0) {
            return input;
        }

        int batchSize = (int) input.size(0);
        int inputChannels = (int) input.size(1);
        int inputHeight = (int) input.size(2);
        int inputWidth = (int) input.size(3);
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;

        INDArray paddedInput = Nd4j.zeros(batchSize, inputChannels, paddedHeight, paddedWidth);
        paddedInput.put(new INDArrayIndex[]{
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(padding, padding + inputHeight),
                NDArrayIndex.interval(padding, padding + inputWidth)
        }, input);
        return paddedInput;
    }

    // Backward pass for the convolutional layer
    public INDArray backward(INDArray error, double learningRate) {
        INDArray inputGradients = Nd4j.zeros(error.shape());
        INDArray[] filterGradients = new INDArray[numFilters];

        for (int f = 0; f < numFilters; f++) {
            filterGradients[f] = Nd4j.zeros(filters[f].shape());

            for (int b = 0; b < error.size(0); b++) {
                for (int i = 0; i < error.size(2); i++) {
                    for (int j = 0; j < error.size(3); j++) {
                        int startX = i * stride;
                        int startY = j * stride;
                        INDArray subMatrix = error.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.all(),
                                NDArrayIndex.point(i),
                                NDArrayIndex.point(j)
                        ).reshape(filterSize, filterSize);

                        filterGradients[f].addi(subMatrix.mul(error.getDouble(b, f, i, j)));
                        inputGradients.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(startX, startX + filterSize),
                                NDArrayIndex.interval(startY, startY + filterSize)
                        ).addi(filters[f].mul(error.getDouble(b, f, i, j)));
                    }
                }
            }

            filters[f].subi(filterGradients[f].mul(learningRate));
        }

        return inputGradients;
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
