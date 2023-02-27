package aufgabe3.neuralNetworkImplementations;

import aufgabe3.activationImplementations.ActivationInterface;
import aufgabe3.activationImplementations.ReLU;
import aufgabe3.costImplementations.CostInterface;
import aufgabe3.inputImplementations.InputInterface;
import aufgabe3.inputImplementations.WeightedAverage;
import aufgabe3.learnData.LayerLearnData;

import java.util.Random;

/**
 * A class to implement a Layer.
 * A Layer is a part of neural Network. Layers are connected with each other to transfer signals.
 * The transfered signal is influenced by the weights/ biases of these connections
 */
public class FullyConnectedLayer implements Layer {

    public int numNodesIn;
    public int numNodesOut;
    public double[] weights;
    public double[] biases;

    // Cost gradient with respect to weights and with respect to biases
    public double[] costGradientW;
    public double[] costGradientB;

    // Used for adding momentum to gradient descent
    public double[] weightVelocities;
    public double[] biasVelocities;
    private ActivationInterface activationFunction;
    private InputInterface inputFunction;

    /**
     * Constructs a new Layer with numNodesIn incomming connections for each node and numNodesOut outgoing
     * connections for each node
     */
    public FullyConnectedLayer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        //each layer is fully connected to the next one. 100 * 100 nodes = 10000 weights
        weights = new double[numNodesIn * numNodesOut];

        costGradientW = new double[weights.length];
        biases = new double[numNodesOut];
        costGradientB = new double[biases.length];

        weightVelocities = new double[weights.length];
        biasVelocities = new double[biases.length];

        setupRandomWeights();

        activationFunction = new ReLU();
        inputFunction = new WeightedAverage();
    }

    /**
     * A method to calculate the outputs of a layer
     * This is done through initially summing up all the inputs for a given node
     * and then normalizing the sum. Each normalized value is then passed through
     * an activation function to generate the output for each node
     * @param inputs
     * @return the layer output activations
     */
    public double[] calculateOutputs(double[] inputs) {
        double[] weightedInputs = inputFunction.calculateOutput(inputs,this);

        //apply activation function
        double[] activations = new double[numNodesOut];
        for (int outputNode = 0; outputNode < numNodesOut; outputNode++) {
            activations[outputNode] = activationFunction.activation(weightedInputs,outputNode);
        }
        return activations;
    }

    /**
     * Same as above, but the process is logged through a LayerLearnData
     * object to use in the learning process of the network
     * @param inputs
     * @param layerLearnData
     * @return
     */
    public double[] calculateOutputs(double[] inputs, LayerLearnData layerLearnData) {
        //The calculated weightedInputs are stored inside the layerlearnData
        inputFunction.calculateOutput(inputs,this,layerLearnData);

        for (int i = 0; i < layerLearnData.activations.length; i++) {
            layerLearnData.activations[i] = activationFunction.activation(layerLearnData.weightedInputs,i);
        }
        return layerLearnData.activations;
    }

    /**
     * Update weights and biases based on gradients
     * resets gradients after the update
     */
    public void applyGradients(double learnRate, double regularization, double momentum){
        double weightDecay = 1 - regularization * learnRate;

        //If the values reach NAN territory we convert it into 0 to avoid running into errors
        for (int i = 0;i<weights.length;i++){
            double weight = weights[i];
            double velocity = weightVelocities[i] * momentum - costGradientW[i] * learnRate;
            weightVelocities[i] = Double.isNaN(velocity) ? 0 : velocity;
            double newWeight = weight * weightDecay + velocity;
            weights[i] = Double.isNaN(newWeight) ? 0 : newWeight;
            costGradientW[i] = 0;
        }

        for (int i = 0;i<biases.length;i++){
            double velocity = biasVelocities[i] * momentum - costGradientB[i] * learnRate;
            biasVelocities[i] = Double.isNaN(velocity) ? 0 : velocity;
            biases[i] += Double.isNaN(velocity) ? 0 : velocity;
            costGradientB[i] = 0;
        }
    }

    /**
     * Update Gradients based on nodeValues (calculated derivatives)
     * Derivatives are calculated based on the derivative function from the given cost function
     */
    public void updateGradients(LayerLearnData layerLearnData){
        for (int nodeOut = 0;nodeOut < numNodesOut;nodeOut++){
            double nodeValue = layerLearnData.nodeValues[nodeOut];
            for (int nodeIn = 0;nodeIn < numNodesIn;nodeIn++){
                //evaluate the partial derivative: cost/weight of current connection
                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we want
                // to calculate the average gradient across all the data in the training batch
                synchronized (new Object()){
                    costGradientW[getFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
                }
            }
        }

        for (int nodeOut = 0;nodeOut < numNodesOut;nodeOut++){
            // Evaluate partial derivative: cost / bias
            double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
            synchronized (new Object()){
                costGradientB[nodeOut] += derivativeCostWrtBias;
            }
        }
    }

    /**
     * Calculate Node values (derivatives)
     */
    public void calculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, CostInterface cost) {
        for (int i = 0; i < layerLearnData.nodeValues.length; i++) {
            double costDerivative = cost.costDerivative(layerLearnData.activations[i], expectedOutputs[i]);
            double activationDeriviative = activationFunction.derivative(layerLearnData.weightedInputs,i);
            layerLearnData.nodeValues[i] = costDerivative * activationDeriviative;
        }
    }

    /**
     * In neural networks, a hidden layer is located between
     * the input and output of the algorithm
     * in which the function applies weights to the
     * inputs and directs them through an activation function as the output
     * @param layerLearnData
     * @param oldLayer
     * @param oldNodeValues
     */
    public void calculateHiddenLayerNodeValues(LayerLearnData layerLearnData, FullyConnectedLayer oldLayer, double[] oldNodeValues) {
        for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                double weightedInputDerivative = oldLayer.getWeight(newNodeIndex, oldNodeIndex);
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            //newNodeValue *= activationFunction.derivative(layerLearnData.weightedInputs, newNodeIndex);
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
        }
    }

    /**
     * A method to calculate random weights and initialize
     */
    public void setupRandomWeights() {
        for (int i = 0; i < weights.length; i++) {
            Random random = new Random();
            double randomNum = random.nextDouble(0, 2);
            weights[i] = randomNum;
        }
    }


    /**
     * A method to calculate the weight of the nodeIn and nodeOut
     *
     * @param nodeIn
     * @param nodeOut
     * @return
     */
    public double getWeight(int nodeIn, int nodeOut) {
        int nodeIndex = nodeOut * numNodesIn + nodeIn;
        return weights[nodeIndex];
    }

    public int getFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex){
        return outputNeuronIndex * numNodesIn + inputNeuronIndex;
    }


    //----------GETTERS/SETTERS---------------

    public int getNumNodesIn() {
        return numNodesIn;
    }

    public int getNumNodesOut() {
        return numNodesOut;
    }

    public double[] getWeights() {
        return weights;
    }

    public ActivationInterface getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationInterface activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void setInputFunction(InputInterface inputFunction) {
        this.inputFunction = inputFunction;
    }
}
