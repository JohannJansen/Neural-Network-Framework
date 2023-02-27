package aufgabe3.costImplementations;

public class CrossEntropy implements CostInterface{
    @Override
    public double costFunction(double[] predictedOutputs, double[] expectedOutputs) {
        double cost = 0;
        for (int i = 0;i < predictedOutputs.length;i++){
            double x = predictedOutputs[i];
            double y = expectedOutputs[i];
            double v = (y == 1) ? -Math.log(x) : -Math.log(1-x);
            cost += Double.isNaN(v) ? 0 : v;
        }
        return cost;
    }

    @Override
    public double costDerivative(double predictedOutput, double expectedOutput) {
        double x = predictedOutput;
        double y = expectedOutput;
        if (x == 0 || x == 1){
            return 0;
        }
        return (-x + y) / (x * (x - 1));
    }
}
