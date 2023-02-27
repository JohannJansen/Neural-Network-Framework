package aufgabe3.costImplementations;

public class MeanSquaredError implements CostInterface {
    @Override
    public double costFunction(double[] predictedOutputs, double[] expectedOutputs) {

        double cost = 0;
        for (int i = 0;i < predictedOutputs.length;i++){
            double error = predictedOutputs[i] - expectedOutputs[i];
            cost += error * error;
        }
        return 0.5 * cost;
    }

    @Override
    public double costDerivative(double predictedOutput, double expectedOutput) {
        return predictedOutput - expectedOutput;
    }
}
