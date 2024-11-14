package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftmaxActivationFunction implements ActivationFunction {

    @Override
    public INDArray activate(INDArray input) {
        // Subtracting the max value for numerical stability
        INDArray inputMinusMax = input.subColumnVector(input.max(1));
        INDArray exp = Transforms.exp(inputMinusMax);
        INDArray sumExp = exp.sum(1);
        return exp.divColumnVector(sumExp);
    }

    @Override
    public INDArray derivative(INDArray output) {
        long batchSize = output.size(0);
        long numClasses = output.size(1);

        // Initialize an empty array for the gradient
        INDArray softmaxGradient = Nd4j.zeros(batchSize, numClasses);

        // Iterate over each sample in the batch
        for (int i = 0; i < batchSize; i++) {
            INDArray outputRow = output.getRow(i);
            INDArray jacobian = computeJacobian(outputRow);
            INDArray sampleGradient = jacobian.mmul(outputRow);
            softmaxGradient.putRow(i, sampleGradient);
        }

        return softmaxGradient;
    }

    private INDArray computeJacobian(INDArray outputRow) {
        long numClasses = outputRow.length();
        INDArray jacobian = Nd4j.zeros(numClasses, numClasses);

        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i == j) {
                    jacobian.putScalar(i, j, outputRow.getDouble(i) * (1 - outputRow.getDouble(i)));
                } else {
                    jacobian.putScalar(i, j, -outputRow.getDouble(i) * outputRow.getDouble(j));
                }
            }
        }
        return jacobian;
    }
}
