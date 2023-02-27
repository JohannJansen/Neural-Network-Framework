package aufgabe3.inputImplementations;

import aufgabe3.neuralNetworkImplementations.FullyConnectedLayer;
import aufgabe3.learnData.LayerLearnData;

public interface InputInterface {
    double[] calculateOutput(double[] inputs, FullyConnectedLayer layer);
    double[] calculateOutput(double[] inputs, FullyConnectedLayer layer, LayerLearnData layerLearnData);
}
