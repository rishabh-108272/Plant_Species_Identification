package org.example;

import org.deeplearning.neuralnetworks.*;
import org.imgscalr.Scalr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.datavec.image.loader.NativeImageLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String RESOURCES_FOLDER_PATH = "C:\\Users\\rishi\\IdeaProjects\\rishabh\\src\\main\\resources\\Output";
    private static final int HEIGHT = 128;
    private static final int WIDTH = 128;
    private static final int CHANNELS = 3; // 3 channels for color images
    private static final int N_OUTCOMES = 7; // Number of classes (adjust as per your dataset)
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    private static final String[] LABELS = {"bougainvillea", "daisy", "frangipani", "hibiscus", "rose", "sunflower", "zinnia"};

    public static void main(String[] args) throws IOException {
        log.info("Starting the training process...");

        File trainData = new File(RESOURCES_FOLDER_PATH + "/train");
        File testData = new File(RESOURCES_FOLDER_PATH + "/test");

        // Initialize the CNN model
        CNNModel cnnModel = new CNNModel(0.0001);

        ActivationFunction RELU = new ReluActivationFunction();
        ActivationFunction SOFTMAX = new SoftmaxActivationFunction();
        // Add layers to the model
        cnnModel.addLayer(new ConvolutionalLayer(2, 3, 1, 1, RELU));
        cnnModel.addLayer(new MaxPoolingLayer(2, 1));
        cnnModel.addLayer(new FullyConnectedLayer(32767, 128, RELU));
        cnnModel.addLayer(new FullyConnectedLayer(128, N_OUTCOMES, SOFTMAX));

        // Training Loop
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            log.info("Epoch " + (epoch + 1) + " started.");
            for (String label : LABELS) {
                File classFolder = new File(trainData, label);
                if (classFolder.isDirectory()) {
                    log.info("Training with class folder: " + classFolder.getName());
                    INDArray input = loadImagesFromDirectory(classFolder, true);
                    System.out.println(input.shapeInfoToString());
                    INDArray output = createLabelsArray(label, (int) input.size(0));
                    System.out.println(output.shapeInfoToString());
                    cnnModel.fit(input, output, 1, BATCH_SIZE);
                }
            }
            log.info("Epoch " + (epoch + 1) + " completed.");
        }

        // Evaluation
        int totalSamples = 0;
        int correctSamples = 0;
        for (String label : LABELS) {
            File classFolder = new File(testData, label);
            if (classFolder.isDirectory()) {
                log.info("Testing with class folder: " + classFolder.getName());
                INDArray input = loadImagesFromDirectory(classFolder, false);
                INDArray output = createLabelsArray(label, (int) input.size(0));
                INDArray predictions = cnnModel.test(input, output);

                totalSamples += input.size(0);
                correctSamples += predictions.eq(output).castTo(Nd4j.defaultFloatingPointType()).sumNumber().intValue();
            }
        }
        log.info("Test accuracy: " + (double) correctSamples / totalSamples);
    }

    private static INDArray loadImagesFromDirectory(File folder, boolean augment) throws IOException {
        List<INDArray> imagesList = new ArrayList<>();
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        for (File imgFile : folder.listFiles()) {
            if (imgFile.isFile()) {
                log.info("Loading image: " + imgFile.getName()); // Log the image name
                BufferedImage originalImage = ImageIO.read(imgFile);
                if (originalImage != null) {
                    BufferedImage processedImage = augment ? augmentImage(originalImage) : originalImage;
                    INDArray image = loader.asMatrix(processedImage);
                    scaler.transform(image);
                    imagesList.add(image);
                }
            }
        }

        // Stack all images into a single INDArray
        return Nd4j.vstack(imagesList);
    }

    private static INDArray createLabelsArray(String label, int numSamples) {
        int labelIndex = -1;
        for (int i = 0; i < LABELS.length; i++) {
            if (LABELS[i].equals(label)) {
                labelIndex = i;
                break;
            }
        }

        if (labelIndex == -1) {
            throw new IllegalArgumentException("Invalid label: " + label);
        }

        INDArray labels = Nd4j.zeros(numSamples, LABELS.length);
        for (int i = 0; i < numSamples; i++) {
            labels.putScalar(new int[]{i, labelIndex}, 1.0);
        }

        return labels;
    }

    private static BufferedImage augmentImage(BufferedImage image) {
        // Example augmentation: flip the image horizontally
        BufferedImage augmentedImage = Scalr.rotate(image, Scalr.Rotation.FLIP_HORZ);
        // Additional augmentation: rotate the image randomly
        double angle = new Random().nextDouble() * 360;
        augmentedImage = rotateImage(augmentedImage, angle);
        return augmentedImage;
    }

    private static BufferedImage rotateImage(BufferedImage image, double angle) {
        double radians = Math.toRadians(angle);
        double sin = Math.abs(Math.sin(radians));
        double cos = Math.abs(Math.cos(radians));
        int width = image.getWidth();
        int height = image.getHeight();
        int newWidth = (int) Math.floor(width * cos + height * sin);
        int newHeight = (int) Math.floor(height * cos + width * sin);

        BufferedImage rotatedImage = new BufferedImage(newWidth, newHeight, image.getType());
        Graphics2D g2d = rotatedImage.createGraphics();
        AffineTransform at = new AffineTransform();
        at.translate((newWidth - width) / 2, (newHeight - height) / 2);
        at.rotate(radians, width / 2, height / 2);
        g2d.setTransform(at);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return rotatedImage;
    }
}
