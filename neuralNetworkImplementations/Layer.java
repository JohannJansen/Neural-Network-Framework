package aufgabe3.neuralNetworkImplementations;

import aufgabe3.costImplementations.CostInterface;
import aufgabe3.learnData.LayerLearnData;

public interface Layer {
    /**
     *
     * @param inputs
     * @return
     */
    double[] calculateOutputs(double[] inputs);
    double[] calculateOutputs(double[] inputs, LayerLearnData layerLearnData);
    void applyGradients(double learnRate, double regularization,double momentum);
    void updateGradients(LayerLearnData layerLearnData);
    void calculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, CostInterface cost);
    void calculateHiddenLayerNodeValues(LayerLearnData layerLearnData, FullyConnectedLayer oldLayer, double[] oldNodeValues);
}
