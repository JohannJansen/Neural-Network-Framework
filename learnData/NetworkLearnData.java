package aufgabe3.learnData;

import aufgabe3.neuralNetworkImplementations.FullyConnectedLayer;

public class NetworkLearnData {
    public LayerLearnData [] layerData;

    /**
     * Constructor
     * @param layers
     */
    public NetworkLearnData (FullyConnectedLayer[] layers){
        layerData = new LayerLearnData[layers.length];
        for (int i = 0; i < layers.length; i++){
            layerData[i] = new LayerLearnData(layers[i]);
        }
    }
}

