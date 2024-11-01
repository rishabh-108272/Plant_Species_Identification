package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.factory.Nd4j;

public class ReluActivationFunction implements ActivationFunction {
    @Override
    public INDArray activate(INDArray input) {
        return Transforms.relu(input, true);
    }

    @Override
    public INDArray derivative(INDArray input) {
        INDArray derivative = Nd4j.zerosLike(input); // Create a zero array of the same shape
        // Set derivative to 1 where input is greater than 0
        derivative = derivative.add(input.gt(0).castTo(derivative.dataType())); // Set to 1 where input > 0
        return derivative;
    }
}
