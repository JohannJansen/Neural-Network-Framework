package aufgabe3.neuralNetworkImplementations;

import aufgabe3.DataPoint;
import aufgabe3.activationImplementations.ActivationInterface;
import aufgabe3.costImplementations.CostInterface;
import aufgabe3.inputImplementations.InputInterface;
import aufgabe3.learnData.NetworkLearnData;

public interface NeuralNetwork {
    double[] calculateOutputs(double [] inputs);
    void learn(DataPoint[] trainingData, double learnrate, double regularization, double momentum);
    void updateGradients(DataPoint data, NetworkLearnData learnData);
    void setCostFunction(CostInterface costFunction);
    void setActivationFunction(ActivationInterface activationFunction,ActivationInterface outputLayerActivation);
    void setInputFunction(InputInterface inputFunction);
}
