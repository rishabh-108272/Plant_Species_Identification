package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class StochasticGradientDescent {
    private double learningRate; // Learning rate for SGD
    private double momentum; // Momentum term
    private INDArray velocity; // Velocity for momentum

    public StochasticGradientDescent(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocity = null; // Initialize velocity to null, will be set during the first update
    }

    public StochasticGradientDescent(double learningRate) {
        this(learningRate, 0.0); // Default to no momentum
    }

    public void update(INDArray weights, INDArray gradients) {
        if (velocity == null) {
            velocity = Nd4j.zeros(weights.shape());
        }

        // Apply momentum
        velocity.muli(momentum).subi(gradients.mul(learningRate));

        // Update weights
        weights.addi(velocity);
    }

    public void updateBias(INDArray bias, INDArray gradients) {
        // Update biases similarly, but without momentum
        bias.subi(gradients.mul(learningRate));
    }
}
