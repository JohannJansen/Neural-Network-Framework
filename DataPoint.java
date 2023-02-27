package aufgabe3;

import java.util.Arrays;

public class DataPoint {
    public double[] inputs;
    public double[] expectedOutput;
    public int label;

    /**
     * Datapoint container holding inputs, the label expected
     * for the inputs and the amount of different labels to classify by
     */
    public DataPoint(double[] inputs, int label,int numLabels) {
        this.inputs = inputs;
        this.label = label;
        this.expectedOutput = new double[numLabels];
        Arrays.fill(expectedOutput,0);
        expectedOutput[label] = 1;
    }
}
