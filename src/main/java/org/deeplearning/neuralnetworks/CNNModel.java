package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

public class CNNModel {
    private List<Object> layers; // A list to hold layers of the model
    private double learningRate; // Learning rate for weight updates

    public CNNModel(double learningRate) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
    }

    // Method to add layers to the model
    public void addLayer(Object layer) {
        layers.add(layer);
    }

    // Forward pass through all layers
    public INDArray forward(INDArray input) {
        INDArray output = input;
        System.out.println("Forward pass");
        for (Object layer : layers) {
            if (layer instanceof ConvolutionalLayer) {
                ConvolutionalLayer convLayer = (ConvolutionalLayer) layer;
                INDArray[] featureMaps = convLayer.forward(output);
                output = concatenateFeatureMaps(featureMaps); // Use optimized concatenation
            } else if (layer instanceof MaxPoolingLayer) {
                MaxPoolingLayer poolLayer = (MaxPoolingLayer) layer;
                output = poolLayer.forward(output);
            } else if (layer instanceof FullyConnectedLayer) {
                // Flatten the output before feeding it into the fully connected layer
                output = flatten(output);
                FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
                output = fcLayer.forward(output);
            }
        }
        System.out.println();
        return output;
    }

    // Helper method to concatenate feature maps into a single INDArray
    private INDArray concatenateFeatureMaps(INDArray[] featureMaps) {
        int numMaps = featureMaps.length;
        long height = featureMaps[0].size(2); // Assuming shape is (batchSize, channels, height, width)
        long width = featureMaps[0].size(3);

        // Create a new INDArray to hold the concatenated result
        INDArray concatenated = Nd4j.create(numMaps, height * width);

        for (int i = 0; i < numMaps; i++) {
            // Directly copy the data from each feature map
            concatenated.putRow(i, featureMaps[i].reshape(1, height * width));
        }

        return concatenated;
    }


    // Method to flatten a 3D tensor to a 1D tensor
    private INDArray flatten(INDArray input) {
        return input.reshape(input.size(0), input.size(1) * input.size(2) * input.size(3));
    }

    public void backward(INDArray output, INDArray expectedOutput) {
        // Calculate deltas for the last layer
        Object lastLayer = layers.get(layers.size() - 1);
        INDArray error = expectedOutput.sub(output); // Compute the error
        System.out.println("Backward pass");
        // Start backpropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Object layer = layers.get(i);

            if (layer instanceof FullyConnectedLayer) {
                FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
                INDArray nextLayerWeights = (i > 0 && layers.get(i - 1) instanceof FullyConnectedLayer)
                        ? ((FullyConnectedLayer) layers.get(i - 1)).getWeights()
                        : null;

                // Backward pass through fully connected layer
                error = fcLayer.backward(error, nextLayerWeights);
                fcLayer.updateWeights(learningRate);
            } else if (layer instanceof ConvolutionalLayer) {
                // Handle backpropagation through convolutional layers
                ConvolutionalLayer convLayer = (ConvolutionalLayer) layer;
                INDArray filterGradients = convLayer.backward(error, learningRate); // Receive a single gradient INDArray

                // Aggregate the gradients
                error = aggregateError(filterGradients); // Update this to handle backpropagation to the previous layer
            } else if (layer instanceof MaxPoolingLayer) {
                // Handle backpropagation through max pooling layers
                MaxPoolingLayer poolLayer = (MaxPoolingLayer) layer;
                error = poolLayer.backward(error); // Ensure this method is implemented in MaxPoolingLayer
            }
        }
        System.out.println();
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
        INDArray output = forward(input);
        // Find the index of the maximum value in each row (the predicted class)
        INDArray predictedClasses = Nd4j.argMax(output, 1);
        INDArray actualClasses = Nd4j.argMax(expectedOutput, 1);

        // Calculate the number of correct predictions
        double correctPredictions = predictedClasses.eq(actualClasses).sumNumber().doubleValue();

        // Calculate accuracy
        double accuracy = correctPredictions / expectedOutput.rows();

        // Return accuracy as an INDArray
        return Nd4j.create(new double[]{accuracy});
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
                System.out.println(batchData);
                INDArray batchLabels = trainingLabels.get(NDArrayIndex.interval(startIndex, endIndex), NDArrayIndex.all());
                System.out.println(batchLabels);
                // Train the model on the current batch
                train(batchData, batchLabels);
            }
            System.out.println();
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }
    }
}
