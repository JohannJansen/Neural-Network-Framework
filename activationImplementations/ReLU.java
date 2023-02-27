package aufgabe3.activationImplementations;

public class ReLU implements ActivationInterface{
    @Override
    public double activation(double[] inputs, int index) {
        return Math.max(0,inputs[index]);
    }

    @Override
    public double derivative(double [] inputs, int index) {
        return inputs[index] >= 0 ? 1 : 0;
    }
}
