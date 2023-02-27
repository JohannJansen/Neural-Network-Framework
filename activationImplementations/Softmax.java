package aufgabe3.activationImplementations;

import java.util.*;

public class Softmax implements ActivationInterface{

    @Override
    public double activation(double[] inputs, int index) {
        double expSum = 0;
        for (double input : inputs) {
            expSum += Math.exp(input);
        }
        return Math.exp(inputs[index]) / expSum;
    }

    @Override
    public double derivative(double[] inputs, int index) {
        double expSum = 0;
        for (int i = 0;i < inputs.length;i++){
            expSum += Math.exp(inputs[i]);
        }
        double ex = Math.exp(inputs[index]);
        return (ex * expSum - ex * ex) / (expSum * expSum);
    }
}
