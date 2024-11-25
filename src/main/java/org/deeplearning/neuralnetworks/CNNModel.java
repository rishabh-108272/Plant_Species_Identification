package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CNNModel {
    private List<Object> layers; // A list to hold layers of the model
    private BatchGradientDescent optimizer; // Optimizer for gradient descent

    public CNNModel(double learningRate) {
        this.layers = new ArrayList<>();
        this.optimizer = new BatchGradientDescent(learningRate);
    }

    // Method to add layers to the model
    public void addLayer(Object layer) {
        layers.add(layer);
    }

    // Forward pass through all layers
    public INDArray forward(INDArray input) {
        INDArray output = input;

//        System.out.println("Forward pass");

        for (Object layer : layers) {
            if (layer instanceof ConvolutionalLayer) {
                ConvolutionalLayer convLayer = (ConvolutionalLayer) layer;
                output = convLayer.forward(output);
//                INDArray[] featureMaps = new INDArray[]{convLayer.forward(output)};
//                System.out.println("Feature Maps:"+ featureMaps[0].shapeInfoToString());
//                output = concatenateFeatureMaps(featureMaps);

//                System.out.println("Output shape after convolutional layer forward pass: " + output.shapeInfoToString());
            } else if (layer instanceof MaxPoolingLayer) {
                MaxPoolingLayer poolLayer = (MaxPoolingLayer) layer;
                output = poolLayer.forward(output);
            } else if (layer instanceof Flatten) {
                Flatten flattend = (Flatten) layer;
                output = flattend.forward(output);
            } else if (layer instanceof FullyConnectedLayer) {
                // Flatten the output before feeding it into the fully connected layer
//                output = flatten(output);
//                System.out.println("Shape of flattened output in 1D array: "+output.shapeInfoToString());
                FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
//                System.out.println("After forward pass of MaxPooling Layer:"+ output.shapeInfoToString());
                output = fcLayer.forward(output);
            }
        }
        System.out.println();
        return output;
    }

    // Method to concatenate feature maps into a single INDArray
    private INDArray concatenateFeatureMaps(INDArray[] featureMaps) {

        int numMaps = featureMaps.length;
        long[] originalShape = featureMaps[0].shape();
        long totalElementsPerMap = featureMaps[0].length();
        long totalElements = numMaps * totalElementsPerMap;

        // Create a new INDArray to hold the concatenated result
        INDArray concatenated = Nd4j.create(totalElements);

        for (int i = 0; i < numMaps; i++) {
            // Directly copy the data from each feature map
            concatenated.put(new INDArrayIndex[]{NDArrayIndex.interval(i * totalElementsPerMap, (i + 1) * totalElementsPerMap)},
                    featureMaps[i].reshape(totalElementsPerMap));
        }

        // Reshape to have all feature maps concatenated along the appropriate dimension
        long[] concatenatedShape = originalShape.clone();
        concatenatedShape[0] = numMaps;  // Adjust the first dimension to reflect the number of feature maps
//        System.out.println("Original Shape:"+ Arrays.toString(concatenatedShape));
//        System.out.println("Shape of concatenated:"+concatenated.shapeInfoToString());
        concatenated = concatenated.reshape(48,3,128,128);
//        System.out.println("Shape of reshaped concatenated:"+concatenated.shapeInfoToString());
        return concatenated;
    }

    // Method to flatten a 4D tensor to a 2D tensor
    private INDArray flatten(INDArray input) {
        return input.reshape(input.size(0), input.size(1) * input.size(2) * input.size(3));
    }

    public void backward(INDArray output, INDArray expectedOutput) {

        INDArray error = expectedOutput.subi(output);

        // List to store gradients of all layers
        List<INDArray> gradients = new ArrayList<>();

        // Start backpropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Object layer = layers.get(i);
//            Object prevlayer = layers.get(i-1);

            if (layer instanceof FullyConnectedLayer) {
                FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
                INDArray nextLayerWeights = (i > 0 && layers.get(i - 1) instanceof FullyConnectedLayer)
                        ? ((FullyConnectedLayer) layers.get(i - 1)).getOutputs()
                        : ((Flatten) layers.get(i-1)).getOutputs();
//                System.out.println("nextLayerweights shape: "+nextLayerWeights.shapeInfoToString());
                // Backward pass through fully connected layer
//                System.out.println("Starting fully connected backward pass");
                INDArray grad = fcLayer.backward(error, nextLayerWeights);
                gradients.add(grad);
                error = grad;
            } else if (layer instanceof ConvolutionalLayer) {
                // Handle backpropagation through convolutional layers
                ConvolutionalLayer convLayer = (ConvolutionalLayer) layer;
                INDArray grad = convLayer.backward(error, optimizer.getLearningRate()); // Receive a single gradient INDArray
                gradients.add(grad);
                error = grad;
//                error = aggregateError(grad); // Update this to handle backpropagation to the previous layer
            } else if (layer instanceof MaxPoolingLayer) {
                // Handle backpropagation through max pooling layers
                MaxPoolingLayer poolLayer = (MaxPoolingLayer) layer;
                error = poolLayer.backward(error); // Ensure this method is implemented in MaxPoolingLayer
            } else if (layer instanceof Flatten) {
                Flatten flatten = (Flatten) layer;
                error = flatten.backward(error);
            }
        }

        // Update weights using gradients and optimizer
//        optimizer.step(getParameters(), gradients);
        System.out.println("Whole Backward pass complete!!!");
    }

    // Method to get all parameters (weights) from the layers
    private List<INDArray> getParameters() {
        List<INDArray> parameters = new ArrayList<>();
        for (Object layer : layers) {
            if (layer instanceof FullyConnectedLayer) {
                parameters.add(((FullyConnectedLayer) layer).getWeights());
            } else if (layer instanceof ConvolutionalLayer) {
                for (INDArray filter : ((ConvolutionalLayer) layer).getFilters()) {
                    parameters.add(filter);
                }
            }
        }
        return parameters;
    }

    private INDArray aggregateError(INDArray filterGradients) {
        // Here, we're assuming that filterGradients is an INDArray containing aggregated gradients from filters
        // Get the dimensions of the gradient
        int height = filterGradients.rows();
        int width = filterGradients.columns();

        // Create an INDArray to hold the aggregated error, initialized to zero
        INDArray aggregatedError = Nd4j.zeros(height, width);

        // Directly add the filter gradients to the aggregated error
        aggregatedError.addi(filterGradients); // Accumulate the gradients

        // Return the aggregated error
        return aggregatedError;
    }

    // Train the model with a single batch of data
    public void train(INDArray input, INDArray expectedOutput) {
        // Forward pass
        INDArray output = forward(input);
        // Backward pass
        backward(output, expectedOutput);
    }

    // Evaluate the model on test data
    public INDArray test(INDArray input, INDArray expectedOutput) {
        return forward(input);
//        INDArray output = forward(input);

        // Find the index of the maximum value in each row (the predicted class)
//        INDArray predictedClasses = Nd4j.argMax(output, 1);
//        INDArray actualClasses = Nd4j.argMax(expectedOutput, 1);
//
//         Calculate the number of correct predictions
//        double correctPredictions = predictedClasses.eq(actualClasses).castTo(DataType.FLOAT).sumNumber().doubleValue();
//
//         Calculate accuracy
//        double accuracy = correctPredictions / expectedOutput.rows();
//
//         Return accuracy as an INDArray
//        return Nd4j.create(new double[]{accuracy});
    }

    // Training loop for multiple epochs
    public void fit(INDArray trainingData, INDArray trainingLabels, int epochs, int batchSize) {
        long numBatches = trainingData.size(0) / batchSize;

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.print("Epoch:" + epoch);
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                System.out.print("=");

                // Extract the current batch of data and labels
                int startIndex = batchIndex * batchSize;
                long endIndex = Math.min(startIndex + batchSize, trainingData.size(0));

                INDArray batchData = trainingData.get(NDArrayIndex.interval(startIndex, endIndex), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
//                System.out.println(batchData);
                System.out.println(batchData.shapeInfoToString());
                INDArray batchLabels = trainingLabels.get(NDArrayIndex.interval(startIndex, endIndex), NDArrayIndex.all());
                System.out.println("Batchlabels shape: "+batchLabels.shapeInfoToString());
//                System.out.println(batchLabels);
                // Train the model on the current batch
                train(batchData, batchLabels);
            }
            System.out.println();
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }
    }
}
