package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActivationFunction {
    INDArray activate(INDArray x);
    INDArray derivative(INDArray x);
}
