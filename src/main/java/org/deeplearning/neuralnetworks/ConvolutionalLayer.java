package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Random;

public class ConvolutionalLayer {
    private INDArray[] filters; // Array of filters
    private int numFilters; // Number of filters
    private int filterSize; // Size of each filter (assumed to be square)
    private int stride; // Stride for convolution
    private int padding; // Padding for convolution
    private ActivationFunction activationFunction;
    INDArray Bias;
    INDArray Weights;
    int []WeightsShape;
    INDArray input;

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

        System.out.println("[CONVOLUTIONAL FORWARD PASS]"+Arrays.toString(input.shape()));

        this.input = input;
        // Intiallize Parameters
        if(WeightsShape == null)
            WeightsShape = new int[]{filterSize,filterSize,(int) input.size(3),numFilters};
        if(Bias == null)
            Bias = Nd4j.rand(numFilters);
        if(Weights == null)
            Weights = Nd4j.rand(WeightsShape);

        // Calculate Output Shape
        int batchSize = (int) input.size(0);
        int inputChannels = (int) input.size(3);
        int inputHeight = (int) input.size(1);
        int inputWidth = (int) input.size(2);
        int outputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;

        // Padd as per given Paddings
        input = Nd4j.pad(input,new int[][]{
                {0,0},
                {padding, padding},
                {padding, padding},
                {0, 0}
        });

        // Initialize output feature maps
        INDArray output = Nd4j.zeros(batchSize, outputHeight, outputWidth, numFilters);

        // Convolution
        for(int b = 0 ; b < batchSize; b++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for(int k = 0; k < numFilters; k++) {
                        int startX = i * stride;
                        int startY = j * stride;

                        INDArray patches = input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(startX, startX + WeightsShape[0]),
                                NDArrayIndex.interval(startY, startY + WeightsShape[1]),
                                NDArrayIndex.all()
                        );
                        INDArray CurrntKernal = this.Weights.get(
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.point(k)
                        );
                        output.putScalar(new int[]{b,i,j,k}, (Double) patches.mul(CurrntKernal).sumNumber());
                    }
                }
            }
        }
//        for (int i = 0; i < outputHeight; i++) {
//            for (int j = 0; j < outputWidth; j++) {
//                int startX = i * stride;
//                int startY = j * stride;
//                patches.get();
//                patches.put(new INDArrayIndex[]{
//                        NDArrayIndex.all(),
//                        NDArrayIndex.all(),
//                        NDArrayIndex.all(),
//                        NDArrayIndex.all(),
//                        NDArrayIndex.point(i),
//                        NDArrayIndex.point(j)
//                }, output.get(NDArrayIndex.all(), NDArrayIndex.all(),
//                        NDArrayIndex.interval(startX, startX + filterSize),
//                        NDArrayIndex.interval(startY, startY + filterSize)));
//            }
//        }
//
//        // Reshape patches to [batchSize * outputHeight * outputWidth, inputChannels * filterSize * filterSize]
//        patches = patches.permute(0, 4, 5, 1, 2, 3).reshape(batchSize * outputHeight * outputWidth, inputChannels * filterSize * filterSize);
//        System.out.println("Reshaped Patches Shape: " + patches.shapeInfoToString());

// Apply convolution filters
//        for (int f = 0; f < numFilters; f++) {
//            // Reshape the filter to [inputChannels * filterSize * filterSize, 1]
//            INDArray filter = filters[f].reshape(inputChannels * filterSize * filterSize, 1);
////            System.out.println("Reshaped Filter Shape: " + filter.shapeInfoToString());
//
//            // Multiply the reshaped filter with the reshaped patches
//            INDArray convolved = patches.mmul(filter);
////            System.out.println("Convolved Shape: "+ convolved.shapeInfoToString());
//            // Reshape the convolved output to [batchSize, outputHeight, outputWidth]
//            convolved = convolved.reshape(batchSize, outputHeight, outputWidth);
////            System.out.println("Convolved reshaped: "+ convolved.shapeInfoToString());
//            convolved = activationFunction.activate(convolved);
//
//            // Place the convolved result in the output array
//            output.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(f), NDArrayIndex.all(), NDArrayIndex.all()}, convolved);
//        }

        output = activationFunction.activate(output);
        System.out.println("[CONVOLUTIONAL FORWARD PASS COMPLETED]"+Arrays.toString(output.shape()));
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
    public INDArray backward(INDArray dZ, double learningRate) {

        System.out.println("[CONVOLUTIONAL BACKWARD PASS]"+Arrays.toString(dZ.shape()));
        long[] inputShape = this.input.shape();
        long[] dZShape = dZ.shape();

        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int inChannels = (int) inputShape[3];
        int dZHeight = (int) dZShape[1];
        int dZWidth = (int) dZShape[2];
        int outChannels = (int) dZShape[3];
        int filterHeight = (int) this.WeightsShape[0];
        int filterWidth = (int) this.WeightsShape[1];

        // Pad the input
        INDArray paddedInput = Nd4j.pad(input, new int[][]{
                {0, 0},               // Batch dimension
                {padding, padding},   // Height padding
                {padding, padding},   // Width padding
                {0, 0}                // Channel dimension
        });

        // Initialize gradients
        INDArray dInput = Nd4j.zerosLike(paddedInput); // Gradient w.r.t input
        INDArray dFilter = Nd4j.zerosLike(this.Weights); // Gradient w.r.t filter
        INDArray dBias = dZ.sum(0, 1, 2); // Gradient w.r.t bias (sum over batch, height, and width)

        // Backpropagation
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < dZHeight; h++) {
                for (int w = 0; w < dZWidth; w++) {
                    int hStart = h * stride;
                    int hEnd = hStart + filterHeight;
                    int wStart = w * stride;
                    int wEnd = wStart + filterWidth;

                    for (int c = 0; c < outChannels; c++) {
                        // Slice input
                        INDArray inputSlice = paddedInput.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        );

                        // Gradient w.r.t. filter
                        dFilter.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c))
                                .addi(inputSlice.mul(dZ.getDouble(b, h, w, c)));

                        // Gradient w.r.t. input
                        dInput.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        ).addi(this.Weights.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c))
                                .mul(dZ.getDouble(b, h, w, c)));
                    }
                }
            }
        }

        // Update weights and biases
        this.Weights.subi(dFilter.mul(learningRate)); // Subtract gradient scaled by learning rate
        this.Bias.subi(dBias.mul(learningRate));     // Subtract gradient scaled by learning rate

        // Remove padding from dInput
        dInput = dInput.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(padding, padding + inputHeight),
                NDArrayIndex.interval(padding, padding + inputWidth),
                NDArrayIndex.all()
        );

        return dInput;
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