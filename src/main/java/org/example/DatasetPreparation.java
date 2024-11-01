//package org.example;
//
//import org.imgscalr.Scalr;
//
//import javax.imageio.ImageIO;
//import java.awt.image.BufferedImage;
//import java.io.File;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.List;
//
//public class DatasetPreparation {
//
//    public static void main(String[] args) {
//        String inputFolderPath = "C:\\Users\\rishi\\IdeaProjects\\Plant_species_Identification\\src\\main\\resources\\flowers"; // Path to the flowers folder
//        String outputFolderPath = "C:\\Users\\rishi\\IdeaProjects\\Plant_species_Identification\\src\\main\\resources\\Output"; // Path to the output folder
//
//        File inputFolder = new File(inputFolderPath);
//        File outputFolder = new File(outputFolderPath);
//
//        File trainFolder = new File(outputFolder, "train");
//        File testFolder = new File(outputFolder, "test");
//
//        double trainTestSplitRatio = 0.8; // 80% for training, 20% for testing
//
//        try {
//            splitAndResizeDataset(inputFolder, trainFolder, testFolder, trainTestSplitRatio, 350, 350);
//            System.out.println("Dataset split and resized successfully!");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    public static void splitAndResizeDataset(File inputFolder, File trainFolder, File testFolder,
//                                             double trainTestSplitRatio, int width, int height) throws IOException {
//        if (!trainFolder.exists()) {
//            trainFolder.mkdirs();
//        }
//        if (!testFolder.exists()) {
//            testFolder.mkdirs();
//        }
//
//        File[] classFolders = inputFolder.listFiles(File::isDirectory);
//
//        if (classFolders != null) {
//            for (File classFolder : classFolders) {
//                File[] images = classFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg")
//                        || name.toLowerCase().endsWith(".jpeg")
//                        || name.toLowerCase().endsWith(".png"));
//
//                if (images != null) {
//                    List<File> imageList = new ArrayList<>();
//                    Collections.addAll(imageList, images);
//                    Collections.shuffle(imageList);
//
//                    int trainSize = (int) (imageList.size() * trainTestSplitRatio);
//
//                    List<File> trainImages = imageList.subList(0, trainSize);
//                    List<File> testImages = imageList.subList(trainSize, imageList.size());
//
//                    System.out.println("Processing class: " + classFolder.getName());
//                    System.out.println("Training images: " + trainImages.size());
//                    System.out.println("Testing images: " + testImages.size());
//
//                    processImages(trainImages, new File(trainFolder, classFolder.getName()), width, height);
//                    processImages(testImages, new File(testFolder, classFolder.getName()), width, height);
//                } else {
//                    System.out.println("No images found in class folder: " + classFolder.getName());
//                }
//            }
//        } else {
//            System.out.println("No class folders found in input folder: " + inputFolder.getAbsolutePath());
//        }
//    }
//
//    private static void processImages(List<File> images, File outputFolder, int width, int height) throws IOException {
//        if (!outputFolder.exists()) {
//            outputFolder.mkdirs();
//        }
//
//        for (File imageFile : images) {
//            try {
//                BufferedImage originalImage = ImageIO.read(imageFile);
//                if (originalImage != null) {
//                    BufferedImage resizedImage = resizeImage(originalImage, width, height);
//                    File outputFile = new File(outputFolder, imageFile.getName());
//                    ImageIO.write(resizedImage, "jpg", outputFile);
//                    System.out.println("Processed and saved: " + outputFile.getAbsolutePath());
//                } else {
//                    System.out.println("Failed to read image: " + imageFile.getAbsolutePath());
//                }
//            } catch (IOException e) {
//                System.out.println("Error processing image: " + imageFile.getAbsolutePath());
//                e.printStackTrace();
//            }
//        }
//    }
//
//    private static BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
//        return Scalr.resize(originalImage, Scalr.Method.QUALITY, Scalr.Mode.AUTOMATIC, width, height);
//    }
//}
