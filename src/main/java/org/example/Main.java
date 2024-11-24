package org.example;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String RESOURCES_FOLDER_PATH = "C:\\Users\\Saurav\\Documents\\Plant_Species_Identification\\src\\main\\resources\\Output";
    private static final int HEIGHT = 128;
    private static final int WIDTH = 128;
    private static final int CHANNELS = 3; // 3 channels for color images
    private static final int N_OUTCOMES = 7; // Number of classes
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 100;
    private static final String[] LABELS = {"bougainvillea", "daisy", "frangipani", "hibiscus", "rose", "sunflower", "zinnia"};

    public static void main(String[] args) throws IOException {
        log.info("Starting the training process...");

        File trainData = new File(RESOURCES_FOLDER_PATH + "/train");
        File testData = new File(RESOURCES_FOLDER_PATH + "/test");

        // Define CNN configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new org.nd4j.linalg.learning.config.Adam(0.0001))
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .nOut(32)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                .build();

        org.deeplearning4j.nn.multilayer.MultiLayerNetwork model = new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        // Training loop
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            log.info("Epoch " + (epoch + 1) + " started.");
            int totalTrainSamples = 0;
            int correctTrainSamples = 0;

            for (String label : LABELS) {
                File classFolder = new File(trainData, label);
                if (classFolder.isDirectory()) {
                    DataSet dataSet = loadImagesAndLabels(classFolder, label);
                    model.fit(dataSet);

                    // Calculate training accuracy
                    INDArray predictions = model.output(dataSet.getFeatures());
                    INDArray labels = dataSet.getLabels();
                    totalTrainSamples += labels.size(0);

                    INDArray correctPredictions = predictions.argMax(1).eq(labels.argMax(1)).castTo(Nd4j.defaultFloatingPointType());
                    correctTrainSamples += correctPredictions.sumNumber().intValue();
                }
            }

            double trainingAccuracy = (double) correctTrainSamples / totalTrainSamples;
            log.info("Epoch " + (epoch + 1) + " completed. Training accuracy: " + trainingAccuracy);
        }

        // Evaluation
        log.info("Evaluating model...");
        int totalSamples = 0;
        int correctSamples = 0;

        for (String label : LABELS) {
            File classFolder = new File(testData, label);
            if (classFolder.isDirectory()) {
                DataSet dataSet = loadImagesAndLabels(classFolder, label);
                INDArray predictions = model.output(dataSet.getFeatures());
                INDArray labels = dataSet.getLabels();
                totalSamples += labels.size(0);

                INDArray correctPredictions = predictions.argMax(1).eq(labels.argMax(1)).castTo(Nd4j.defaultFloatingPointType());
                correctSamples += correctPredictions.sumNumber().intValue();
            }
        }
        log.info("Test accuracy: " + (double) correctSamples / totalSamples);
    }

    private static DataSet loadImagesAndLabels(File folder, String label) throws IOException {
        List<INDArray> imagesList = new ArrayList<>();
        List<INDArray> labelsList = new ArrayList<>();
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        int labelIndex = -1;
        for (int i = 0; i < LABELS.length; i++) {
            if (LABELS[i].equals(label)) {
                labelIndex = i;
                break;
            }
        }

        for (File imgFile : folder.listFiles()) {
            if (imgFile.isFile()) {
                BufferedImage image = ImageIO.read(imgFile);
                if (image != null) {
                    INDArray input = loader.asMatrix(image);
                    scaler.transform(input);
                    imagesList.add(input);
                    INDArray output = Nd4j.zeros(1, LABELS.length);
                    output.putScalar(new int[]{0, labelIndex}, 1.0);
                    labelsList.add(output);
                }
            }
        }

        return new DataSet(Nd4j.vstack(imagesList), Nd4j.vstack(labelsList));
    }
}
