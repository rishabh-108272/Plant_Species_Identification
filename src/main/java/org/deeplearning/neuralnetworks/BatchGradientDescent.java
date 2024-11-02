package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

public class BatchGradientDescent {

    private double learningRate;  // Learning rate for the gradient descent

    public BatchGradientDescent(double learningRate) {
        this.learningRate = learningRate;
    }

    // Method to perform a step of gradient descent
    public void step(List<INDArray> parameters, List<INDArray> gradients) {
        if (parameters.size() != gradients.size()) {
            throw new IllegalArgumentException("The size of parameters and gradients must be the same.");
        }

        // Update each parameter by subtracting the gradient scaled by the learning rate
        for (int i = 0; i < parameters.size(); i++) {
            INDArray param = parameters.get(i);
            INDArray grad = gradients.get(i);
            param.subi(grad.mul(learningRate));  // Update parameter in-place
        }
    }

    public double getLearningRate() {
        return learningRate;
    }
}
