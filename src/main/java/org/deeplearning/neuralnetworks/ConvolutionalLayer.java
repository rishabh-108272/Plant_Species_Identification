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
            filters[i] = Nd4j.rand(new int[] {1, filterSize, filterSize}).subi(0.5); // Initialize filters with values between -0.5 and 0.5
        }
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

        // Pad the input
        INDArray paddedInput = padInput(input, padding);

        // Extract patches from the input
        INDArray patches = Nd4j.create(new int[]{batchSize, inputChannels, filterSize, filterSize, outputHeight, outputWidth});

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

        // Reshape patches to apply convolution
//        patches = patches.reshape(batchSize, inputChannels * filterSize * filterSize, outputHeight * outputWidth);

        // Apply convolution filters
        for (int f = 0; f < numFilters; f++) {
            INDArray filter = filters[f].reshape(1,batchSize);
            INDArray convolved = filter.mmul(patches);
            convolved = convolved.reshape(batchSize, outputHeight, outputWidth);
            convolved = activationFunction.activate(convolved);
            output.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(f), NDArrayIndex.all(), NDArrayIndex.all()}, convolved);
        }

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
