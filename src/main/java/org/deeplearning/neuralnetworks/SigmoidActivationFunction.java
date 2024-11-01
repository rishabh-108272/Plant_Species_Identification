package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public INDArray activate(INDArray x) {
        // Compute the sigmoid function element-wise: 1 / (1 + exp(-x))
        INDArray negatedX = x.neg(); // Negate the input array

        // Initialize the output array
        INDArray output = Nd4j.create(x.shape());

        // Compute the exp(-x) for each element and fill the output array
        for (int i = 0; i < negatedX.length(); i++) {
            output.putScalar(i, Math.exp(negatedX.getDouble(i))); // Use Math.exp for element-wise calculation
        }

        INDArray one = Nd4j.create(new double[]{1.0}); // Create a scalar one
        return one.div(one.add(output)); // Compute 1 / (1 + exp(-x))
    }


    @Override
    public INDArray derivative(INDArray x) {
        // Calculate the derivative of the sigmoid function: sigmoid * (1 - sigmoid)
        INDArray sigmoid = activate(x); // Get the sigmoid output
        return sigmoid.mul(sigmoid.rsub(1)); // Compute sigmoid * (1 - sigmoid) using rsub for 1 - sigmoid
    }
}
