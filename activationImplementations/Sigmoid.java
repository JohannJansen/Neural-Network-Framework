package aufgabe3.activationImplementations;

public class Sigmoid implements ActivationInterface{
    @Override
    public double activation(double [] inputs, int index) {
        return 1.0 / (1 + Math.exp(-inputs[index]));
    }

    @Override
    public double derivative(double [] inputs, int index) {
        double a = activation(inputs, index);
        return a * (1 - a);
    }
}
