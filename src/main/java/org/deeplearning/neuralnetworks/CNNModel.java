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

        for (Object layer : layers) {
            if (layer instanceof ConvolutionalLayer) {
                ConvolutionalLayer convLayer = (ConvolutionalLayer) layer;
                INDArray[] featureMaps = convLayer.forward(output);
                output = concatenateFeatureMaps(featureMaps); // Concatenate or process the feature maps
            } else if (layer instanceof MaxPoolingLayer) {
                MaxPoolingLayer poolLayer = (MaxPoolingLayer) layer;
                output = poolLayer.forward(output);
            } else if (layer instanceof FullyConnectedLayer) {
                FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
                output = fcLayer.forward(output);
            }
        }
        return output;
    }

    // Helper method to concatenate feature maps into a single INDArray
    private INDArray concatenateFeatureMaps(INDArray[] featureMaps) {
        // Assuming feature maps are of the same size
        int height = featureMaps[0].rows();
        int width = featureMaps[0].columns();
        int numMaps = featureMaps.length;

        // Create a new INDArray to hold the concatenated result
        INDArray concatenated = Nd4j.create(numMaps, height * width);

        for (int i = 0; i < numMaps; i++) {
            concatenated.putRow(i, featureMaps[i].reshape(1, height * width));
        }

        return concatenated;
    }


    public void backward(INDArray expectedOutput) {
        // Calculate deltas for the last layer
        Object lastLayer = layers.get(layers.size() - 1);
        INDArray output = ((FullyConnectedLayer) lastLayer).getOutputs();
        INDArray error = expectedOutput.sub(output); // Compute the error

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
        backward(expectedOutput);
    }

    // Evaluate the model on test data
    public double test(INDArray input, INDArray expectedOutput) {
        INDArray output = forward(input);
        // Calculate accuracy or other metrics
        double correctPredictions = output.eq(expectedOutput).sumNumber().doubleValue();
        return correctPredictions / expectedOutput.rows(); // Return accuracy
    }

    // Training loop for multiple epochs
    public void fit(INDArray trainingData, INDArray trainingLabels, int epochs, int batchSize) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingData.rows(); i += batchSize) {
                INDArray batchData = trainingData.get(NDArrayIndex.interval(i, Math.min(i + batchSize, trainingData.rows())), NDArrayIndex.all());
                INDArray batchLabels = trainingLabels.get(NDArrayIndex.interval(i, Math.min(i + batchSize, trainingLabels.rows())), NDArrayIndex.all());
                train(batchData, batchLabels);
            }
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }
    }
}
