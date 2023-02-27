package aufgabe3.costImplementations;


public interface CostInterface {
    double costFunction(double[] predictedOutputs, double[] expectedOutputs);

    double costDerivative(double predictedOutput, double expectedOutput);
}
