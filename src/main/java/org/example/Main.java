package org.example;

import org.datavec.api.split.FileSplit;
import org.deeplearning.neuralnetworks.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import org.nd4j.linalg.factory.Nd4j;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String RESOURCES_FOLDER_PATH = "C:\\Users\\rishi\\IdeaProjects\\Plant_species_Identification\\src\\main\\resources\\Output";
    private static final int HEIGHT = 325;
    private static final int WIDTH = 325;
    private static final int CHANNELS = 3; // 3 channels for color images
    private static final int N_OUTCOMES = 7; // Number of classes (adjust as per your dataset)
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    private static DataSetIterator getDataSetIterator(File folder, int batchSize, int numLabels) throws IOException {
        FileSplit fileSplit = new FileSplit(folder, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        recordReader.initialize(fileSplit);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIterator.setPreProcessor(scaler);
        return dataSetIterator;
    }

    public static void main(String[] args) throws IOException {
        log.info("Starting the training process...");

        File trainData = new File(RESOURCES_FOLDER_PATH + "/train");
        File testData = new File(RESOURCES_FOLDER_PATH + "/test");

        DataSetIterator trainIterator = null;
        DataSetIterator testIterator = null;
        try {
            trainIterator = getDataSetIterator(trainData, BATCH_SIZE, N_OUTCOMES);
            testIterator = getDataSetIterator(testData, BATCH_SIZE, N_OUTCOMES);
        } catch (IOException e) {
            log.error("Failed to initialize data iterators", e);
            return;
        }

        // Initialize the CNN model
        CNNModel cnnModel = new CNNModel(0.01);

        ActivationFunction RELU= new ReluActivationFunction();
        ActivationFunction SOFTMAX= new SoftmaxActivationFunction();
        // Add layers to the model
        cnnModel.addLayer(new ConvolutionalLayer(32, 3, 1, 0,RELU)); // Convolutional layer example
        cnnModel.addLayer(new MaxPoolingLayer(2, 2)); // Max pooling layer
        cnnModel.addLayer(new FullyConnectedLayer(32 * 162 * 162, 128, RELU)); // Fully connected layer example
        cnnModel.addLayer(new FullyConnectedLayer(128, N_OUTCOMES, SOFTMAX)); // Output layer

        // Training Loop
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            while (trainIterator.hasNext()) {
                DataSet dataSet = trainIterator.next();
                try {
                    cnnModel.train(dataSet.getFeatures(), dataSet.getLabels());
                } catch (Exception e) {
                    log.error("Error during training at epoch " + epoch, e);
                    return;
                }
            }
            log.info("Epoch " + (epoch + 1) + " completed.");
            trainIterator.reset();
        }

        // Evaluation
        int totalSamples = 0;
        int correctSamples = 0;
        while (testIterator.hasNext()) {
            INDArray features = testIterator.next().getFeatures();
            INDArray labels = testIterator.next().getLabels();
            INDArray output = cnnModel.forward(features);

            totalSamples += labels.size(0);
            correctSamples += output.eq(labels).castTo(Nd4j.defaultFloatingPointType()).sumNumber().intValue();
        }
        log.info("Test accuracy: " + (double) correctSamples / totalSamples);
    }
}
