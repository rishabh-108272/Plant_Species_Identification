package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MaxPoolingLayer {
    private int poolSize; // Size of the pooling window (assumed to be square)
    private int stride; // Stride for pooling

    public MaxPoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    // Forward pass for the max pooling layer
    public INDArray forward(INDArray input) {
        int inputHeight = input.rows();
        int inputWidth = input.columns();
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        INDArray output = Nd4j.create(outputHeight, outputWidth);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                int startX = i * stride;
                int startY = j * stride;
                INDArray subMatrix = input.get(NDArrayIndex.interval(startX, startX + poolSize), NDArrayIndex.interval(startY, startY + poolSize));
                double maxVal = subMatrix.maxNumber().doubleValue();
                output.putScalar(i, j, maxVal);
            }
        }
        return output;
    }
    public INDArray backward(INDArray upstreamGradient) {
        int inputHeight = (upstreamGradient.rows() - 1) * stride + poolSize; // Calculate the input height
        int inputWidth = (upstreamGradient.columns() - 1) * stride + poolSize; // Calculate the input width
        INDArray inputGradient = Nd4j.zeros(inputHeight, inputWidth); // Initialize gradient for the input

        // Backward pass for max pooling
        for (int i = 0; i < upstreamGradient.rows(); i++) {
            for (int j = 0; j < upstreamGradient.columns(); j++) {
                // Determine the position of the maximum value in the original input
                int startX = i * stride;
                int startY = j * stride;

                // Get the submatrix that was pooled
                INDArray subMatrix = inputGradient.get(NDArrayIndex.interval(startX, startX + poolSize), NDArrayIndex.interval(startY, startY + poolSize));

                // Find the max value's index in the subMatrix
                double maxVal = subMatrix.maxNumber().doubleValue();
                int[] maxIndices = new int[2]; // Store the indices of the max value

                // Iterate through the subMatrix to find the max value and its index
                for (int m = 0; m < poolSize; m++) {
                    for (int n = 0; n < poolSize; n++) {
                        if (subMatrix.getDouble(m, n) == maxVal) {
                            maxIndices[0] = startX + m; // Row index in input gradient
                            maxIndices[1] = startY + n; // Column index in input gradient
                            break;
                        }
                    }
                }

                // Assign the upstream gradient value to the position of the max value
                inputGradient.putScalar(maxIndices[0], maxIndices[1], upstreamGradient.getDouble(i, j));
            }
        }

        return inputGradient; // Return the gradient for the input
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }
}
