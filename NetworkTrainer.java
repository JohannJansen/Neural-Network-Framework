package aufgabe3;

import aufgabe3.activationImplementations.ReLU;
import aufgabe3.activationImplementations.Sigmoid;
import aufgabe3.activationImplementations.Softmax;
import aufgabe3.costImplementations.CrossEntropy;
import aufgabe3.inputImplementations.Sum;
import aufgabe3.inputImplementations.WeightedAverage;
import aufgabe3.mnistReadClasses.MnistDataReader;
import aufgabe3.neuralNetworkImplementations.FullyConnectedNeuralNetwork;
import aufgabe3.neuralNetworkImplementations.NeuralNetwork;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

public class NetworkTrainer {
    public float trainingSplit = 0.8f;

    public DataPoint[] allData,trainingData,validationData;
    public DataPoint[][] trainingsbatches;
    public NeuralNetwork neuralNetwork;
    double inititalLearnRate = 1f;
    double currentLearnRate = inititalLearnRate;
    double learnRateDecay = 0;
    //Regularization is used to determine the weight decay. A higher number means the current weight is less impactful
    //in the calculation of the new value
    double regularization = 0.1;
    //Momentum defines how much the previous changes to a weight (its previous velocity) impacts the new velocity
    //a lower number indicates less of an impact on the new velocity
    double momentum = 0.4;
    public int epochs;

    /**
     * Constructs a new Network Trainer. This is used to train a neuralNetwork and test the effectiveness afterwards
     * @throws IOException
     */
    public NetworkTrainer() throws IOException {
        this.neuralNetwork = new FullyConnectedNeuralNetwork(new int[]{784, 100, 10});
        neuralNetwork.setActivationFunction(new ReLU(),new Softmax());
        neuralNetwork.setCostFunction(new CrossEntropy());
        neuralNetwork.setInputFunction(new WeightedAverage());
        loadData();
        this.epochs = 1000;
        trainingsbatches = new DataPoint[epochs][trainingData.length / epochs - 1];
    }

    /**
     * Loads the Minset DataSet and stores each Image as a Datapoint containing an Array with grayscalevalues for
     * each pixel and the label associated with the image
     * @throws IOException
     */
    public void loadData() throws IOException {
        //allData = MnistDataReader.readDataIntoDataPoints("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        trainingData = MnistDataReader.readDataIntoDataPoints("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        validationData = MnistDataReader.readDataIntoDataPoints("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    }

    /**
     * Splits the trainingData into smaller sized batches based on the number of epochs specified
     * @param trainingData
     * @param epochs
     */
    public void createTrainingBatches(DataPoint[] trainingData,int epochs){
        int index = trainingData.length / epochs - 1;
        int currentIndex = 0;
        for (int i = 0;i<epochs;i++){
            trainingsbatches[i] = Arrays.copyOfRange(trainingData,currentIndex,currentIndex+index);
            currentIndex += index;
        }
    }

    /**
     * Trains the Network through epochs steps of learning
     * Throughout this process the learnrate steadily declines based on the learnratedecay
     */
    public void trainNetwork(){
        createTrainingBatches(trainingData,epochs);
        for (int i = 0;i<epochs;i++){
            neuralNetwork.learn(trainingsbatches[i],currentLearnRate,regularization,momentum);
            currentLearnRate = (1.0 / (1.0 + learnRateDecay * i) * inititalLearnRate);
        }

    }

    /**
     * Evaluates the effectiveness of the trained network
     * The first 3 outputs are labels with the respective estimations
     * Afterwards a summary of right/wrong classifications of the trainingsdata is printed
     */
    public void evaluate(){
        DecimalFormat df = new DecimalFormat("0.00");


        for (int i = 0;i<3;i++){
            double[] output = neuralNetwork.calculateOutputs(validationData[i].inputs);
            System.out.println("label" + validationData[i].label);
            for (int n = 0;n < output.length;n++){
                System.out.println("Number: " + n + " was detected with a certanty of: " + df.format(output[n]));
            }
        }

        System.out.println("Results for training Data:");
        runNetworkOnData(trainingData);
        System.out.println("Results for validation Data:");
        runNetworkOnData(validationData);
    }

    public void runNetworkOnData(DataPoint[] data){
        int numberRight = 0;
        int numberWrong = 0;
        for (int i = 0;i< data.length;i++){

            double[] output = neuralNetwork.calculateOutputs(data[i].inputs);
            double maxval = Double.MIN_VALUE;
            int indexMaxVal = -1;
            for (int n = 0;n < output.length;n++){
                if (output[n] > maxval) {
                    maxval = output[n];
                    indexMaxVal = n;
                }
            }
            if (indexMaxVal == data[i].label){
                numberRight++;
            }
            else {
                numberWrong++;
            }

        }
        System.out.println("Number of right classified: " + numberRight + " Number of wrong classified: " + numberWrong);
    }

    public static void main(String[] args) throws IOException {
        long starttime = System.currentTimeMillis();
        NetworkTrainer networkTrainer = new NetworkTrainer();
        networkTrainer.trainNetwork();
        networkTrainer.evaluate();
        System.out.println("Total runtime in Millis: " + (int)(System.currentTimeMillis() - starttime));
    }
}
