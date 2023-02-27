package aufgabe3.inputImplementations;

import aufgabe3.learnData.LayerLearnData;
import aufgabe3.neuralNetworkImplementations.FullyConnectedLayer;

public class Sum implements InputInterface{
    @Override
    public double[] calculateOutput(double[] inputs, FullyConnectedLayer layer) {
        double[] weightedInputs = new double[layer.numNodesOut];

        for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++){
            double weightedInput = layer.biases[nodeOut];

            for (int nodeIn = 0;nodeIn < layer.numNodesIn;nodeIn++){
                weightedInput += inputs[nodeIn] * layer.getWeight(nodeIn,nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }
        return weightedInputs;
    }

    @Override
    public double[] calculateOutput(double[] inputs, FullyConnectedLayer layer, LayerLearnData layerLearnData) {
        layerLearnData.inputs = inputs;

        for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++) {
            //insert bias here
            double weightedInput = layer.biases[nodeOut];

            for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * layer.getWeight(nodeIn, nodeOut);
            }
            layerLearnData.weightedInputs[nodeOut] = weightedInput;
        }
        return layerLearnData.weightedInputs;
    }
}
