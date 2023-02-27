package aufgabe3.learnData;

import aufgabe3.neuralNetworkImplementations.FullyConnectedLayer;

public class LayerLearnData {
    public double[] inputs;
    public double[] weightedInputs;
    public double[] activations;
    public double[] nodeValues;

    public LayerLearnData(FullyConnectedLayer layer)
    {
        weightedInputs = new double[layer.getNumNodesOut()];
        activations = new double[layer.getNumNodesOut()];
        nodeValues = new double[layer.getNumNodesOut()];
    }
}
