package org.deeplearning.neuralnetworks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class Layer {
    private ArrayList<Neuron> neurons;

    public Layer(int numNeurons, int numInputsPerNeuron, ActivationFunction activationFunction) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new Neuron(numInputsPerNeuron, activationFunction));
        }
    }

    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }

    public INDArray getOutputs(INDArray inputs) {
        INDArray outputs = Nd4j.zeros(neurons.size(), 1); // Initialize outputs as a column vector
        for (int i = 0; i < neurons.size(); i++) {
            INDArray neuronOutput = neurons.get(i).activate(inputs); // Activate each neuron
            outputs.putScalar(i, neuronOutput.getDouble(0)); // Store the scalar output from INDArray
        }
        return outputs;
    }
}
