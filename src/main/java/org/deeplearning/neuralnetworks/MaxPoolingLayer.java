package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Array;
import java.util.Arrays;

public class MaxPoolingLayer {
    private int poolSize; // Size of the pooling window (assumed to be square)
    private int stride; // Stride for pooling
    INDArray input;

    public MaxPoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    // Forward pass for the max pooling layer
    public INDArray forward(INDArray input) {

        System.out.println("[MAXIMUM POOLING FORWARD PASS]"+ Arrays.toString(input.shape()));

        this.input = input;
        int batchSize = (int) input.size(0);
        int inputHeight = (int) input.size(1);
        int inputWidth = (int) input.size(2);
        int channels = (int) input.size(3);

        // Calculate output dimensions
        int outputHeight = ((inputHeight - poolSize) / stride) + 1;
        int outputWidth = ((inputWidth - poolSize) / stride) + 1;

        // Initialize output array with 4D shape [batchSize, channels, outputHeight, outputWidth]
        INDArray output = Nd4j.create(batchSize, outputHeight, outputWidth, channels);

        // Perform max pooling
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int k = 0; k < channels; k++) {
                        int startX = i * stride;
                        int startY = j * stride;

                        // Get pooling region
                        INDArray subMatrix = input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(startX, startX + poolSize),
                                NDArrayIndex.interval(startY, startY + poolSize),
                                NDArrayIndex.point(k)
                        );

                        output.putScalar(new int[]{b, i, j, k}, (double) subMatrix.maxNumber());
                    }
                }
            }
        }

        System.out.println("[MAXIMUM POOLING FORWARD PASS COMPLETED]"+Arrays.toString(output.shape()));
        return output;
    }

    INDArray backward(INDArray dZ) {

        System.out.println("[MAXIMUM POOLING BACKWARD PASS]"+Arrays.toString(dZ.shape()));

        long[] inputShape = this.input.shape();
        INDArray dP = Nd4j.zeros(inputShape);
        for (int b = 0; b < inputShape[0]; b++) {
            for (int h = 0; h < inputShape[1] - this.poolSize/this.stride + 1; h++) {
                for (int w = 0; w < inputShape[2] - this.poolSize/this.stride + 1; w++) {
                    for (int c = 0; c < inputShape[3]; c++) {

                        int hStart = h * this.stride;
                        int wStart = w * this.stride;

                        int hEnd = (int) Math.min(hStart + this.poolSize, inputShape[1]);
                        int wEnd = (int) Math.min(wStart + this.poolSize, inputShape[2]);

                        INDArray window = this.input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        );


                        int[] maxIndex = Nd4j.argMax(window.reshape(1, -1), 1).toIntVector();
                        int maxH = maxIndex[0] / (int) this.poolSize;
                        int maxW = maxIndex[0] % (int) this.poolSize;

                        // Assign gradient to corresponding input index
                        double gradient = dZ.getDouble(b, h , w , c);
                        // Ensure the indices are within the bounds of the input array
                        int hIndex = (int) Math.min((long) h * this.stride + maxH, inputShape[1] - 1);
                        int wIndex = (int) Math.min((long) w * this.stride + maxW, inputShape[2] - 1);

                        dP.putScalar(new int[]{b, hIndex, wIndex, c}, gradient);
                    }
                }
            }
        }
        return dP;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }
}
