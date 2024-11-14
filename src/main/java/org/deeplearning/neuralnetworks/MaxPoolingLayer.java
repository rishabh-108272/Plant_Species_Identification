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
        System.out.println("Maxpooling Layer input shape: " + input.shapeInfoToString());

        int batchSize = (int) input.size(0);
        int channels = (int) input.size(1);
        int inputHeight = (int) input.size(2);
        int inputWidth = (int) input.size(3);

        // Calculate output dimensions
        int outputHeight = ((inputHeight - poolSize) / stride) + 1;
        int outputWidth = ((inputWidth - poolSize) / stride) + 1;

        // Initialize output array with 4D shape [batchSize, channels, outputHeight, outputWidth]
        INDArray output = Nd4j.create(batchSize, channels, outputHeight, outputWidth);

        // Perform max pooling
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        int startX = i * stride;
                        int startY = j * stride;

                        // Get pooling region
                        INDArray subMatrix = input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.point(c),
                                NDArrayIndex.interval(startX, startX + poolSize),
                                NDArrayIndex.interval(startY, startY + poolSize)
                        );

                        // Find the maximum value in the subMatrix
                        double maxVal = subMatrix.maxNumber().doubleValue();
                        output.putScalar(new int[]{b, c, i, j}, maxVal);
                    }
                }
            }
        }

        System.out.println("MaxPooling layer forward pass completed!");
        System.out.println("MaxPooling Layer output shape: " + output.shapeInfoToString());
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
