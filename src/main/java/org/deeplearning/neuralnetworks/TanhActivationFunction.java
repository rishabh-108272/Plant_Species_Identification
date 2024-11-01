package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TanhActivationFunction implements ActivationFunction {
    @Override
    public INDArray activate(INDArray input) {
        return Transforms.tanh(input, true);
    }

    @Override
    public INDArray derivative(INDArray input) {
        INDArray tanhValues = activate(input); // Get the tanh values
        return Nd4j.ones(tanhValues.shape()).sub(tanhValues.mul(tanhValues)); // Derivative: 1 - tanh^2
    }
}
