package aufgabe3.neuralNetworkImplementations;

import aufgabe3.DataPoint;
import aufgabe3.activationImplementations.ActivationInterface;
import aufgabe3.costImplementations.CostInterface;
import aufgabe3.costImplementations.MeanSquaredError;
import aufgabe3.inputImplementations.InputInterface;
import aufgabe3.learnData.LayerLearnData;
import aufgabe3.learnData.NetworkLearnData;
import aufgabe3.neuralNetworkImplementations.FullyConnectedLayer;

public class FullyConnectedNeuralNetwork implements NeuralNetwork {
    private FullyConnectedLayer[] layers;
    private int[] layerSizes;
    private CostInterface costFunction;
    private NetworkLearnData[] networkLearnData;

    /**
     * Constructs a new Neuralnetwork based on the given layersizes. The given arrays first and last
     * entries are the input/output layers sizes
     * @param layerSizes
     */
    public FullyConnectedNeuralNetwork(int[] layerSizes){

        this.layerSizes = layerSizes;

        layers = new FullyConnectedLayer[layerSizes.length - 1];
        for (int i = 0;i<layers.length;i++){
            layers[i] = new FullyConnectedLayer(layerSizes[i],layerSizes[i+1]);
        }

        costFunction = new MeanSquaredError();
    }

    /**
     * Calculates the output of the whole network for a given input
     * This is done by calculating the output for each layer based on the output of
     * the previous layer/input
     * @param inputs
     * @return
     */
    public double[] calculateOutputs(double [] inputs){
        for (FullyConnectedLayer l: layers) {
            inputs = l.calculateOutputs(inputs);
        }
        return inputs;
    }

    /**
     * Network learns with given training data
     */
    public void learn(DataPoint[] trainingData, double learnrate, double regularization, double momentum){
        if (networkLearnData == null){
            networkLearnData = new NetworkLearnData[trainingData.length];
            for (int i = 0;i < networkLearnData.length;i++){
                networkLearnData[i] = new NetworkLearnData(layers);
            }
        }

        //parallel processing of batches
        for (int i = 0;i < networkLearnData.length;i++){
            int finalI = i;
            Thread batchThread = new Thread(() -> updateGradients(trainingData[finalI],networkLearnData[finalI]));
            batchThread.start();
            //updateGradients(trainingData[i],networkLearnData[i]);
        }

        for (int i = 0;i < layers.length;i++){
            layers[i].applyGradients(learnrate / trainingData.length ,regularization,momentum);
        }
    }

    /**
     * Updates the gradients for each layer based on the given Datapoint
     * @param data
     * @param learnData
     */
    public void updateGradients(DataPoint data, NetworkLearnData learnData){
        double[] inputsToNextLayer = data.inputs;

        for (int i = 0;i<layers.length;i++){
            inputsToNextLayer = layers[i].calculateOutputs(inputsToNextLayer,learnData.layerData[i]);
        }

        //Backpropagation
        int outputLayerIndex = layers.length - 1;
        FullyConnectedLayer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];
        //update output layer gradients
        outputLayer.calculateOutputLayerNodeValues(outputLearnData, data.expectedOutput, costFunction);
        outputLayer.updateGradients(outputLearnData);
        //repeat process for every hidden layer
        for (int i = outputLayerIndex - 1;i >= 0;i--){
            LayerLearnData layerLearnData = learnData.layerData[i];
            FullyConnectedLayer hiddenLayer = layers[i];

            hiddenLayer.calculateHiddenLayerNodeValues(layerLearnData,layers[i+1],learnData.layerData[i+1].nodeValues);
            hiddenLayer.updateGradients(layerLearnData);
        }
    }

    public void setCostFunction(CostInterface costFunction){
        this.costFunction = costFunction;
    }

    public void setActivationFunction(ActivationInterface activationFunction){
        setActivationFunction(activationFunction,activationFunction);
    }

    public void setActivationFunction(ActivationInterface activationFunction,ActivationInterface outputLayerActivation){
        for (int i = 0;i < layers.length - 1;i++){
            layers[i].setActivationFunction(activationFunction);
        }
        layers[layers.length - 1].setActivationFunction(outputLayerActivation);
    }

    public void setInputFunction(InputInterface inputFunction){
        for (int i = 0;i<layers.length;i++){
            layers[i].setInputFunction(inputFunction);
        }
    }

}
