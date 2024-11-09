package RN.transformer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Transformer implements Serializable {
	
    private static final long serialVersionUID = 1L;
    
    private Encoder encoder;
    private Decoder decoder;
    private CustomAdamOptimizer optimizer;
    // Autres attributs du Transformer...

    public Transformer(/* paramètres du constructeur */) {
        // Initialisation du Transformer...
    }

    // Méthodes existantes du Transformer...

    public void saveState(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            // Sauvegarder l'état de l'encodeur et du décodeur
            oos.writeObject(encoder);
            oos.writeObject(decoder);
            
            // Sauvegarder l'état de l'optimiseur
            oos.writeObject(optimizer.getCurrentStep());
            oos.writeObject(optimizer.getEpoch());
            oos.writeObject(optimizer.getLearningRate());
            
            // Sauvegarder les paramètres du modèle
            List<INDArray> parameters = getParameters();
            oos.writeObject(parameters.size());
            for (INDArray param : parameters) {
                oos.writeObject(param);
            }
        }
    }

    public void loadState(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            // Charger l'état de l'encodeur et du décodeur
            this.encoder = (Encoder) ois.readObject();
            this.decoder = (Decoder) ois.readObject();
            
            // Charger l'état de l'optimiseur
            int currentStep = (int) ois.readObject();
            int epoch = (int) ois.readObject();
            float learningRate = (float) ois.readObject();
            optimizer.setCurrentStep(currentStep);
            optimizer.setEpoch(epoch);
            optimizer.setLearningRate(learningRate);
            
            // Charger les paramètres du modèle
            int numParams = (int) ois.readObject();
            List<INDArray> parameters = getParameters();
            for (int i = 0; i < numParams; i++) {
                INDArray param = (INDArray) ois.readObject();
                parameters.get(i).assign(param);
            }
        }
    }
    


    private List<INDArray> getParameters() {
        // Méthode pour obtenir tous les paramètres du modèle
        // Combinez les paramètres de l'encodeur et du décodeur
        List<INDArray> params = encoder.getParameters();
        params.addAll(decoder.getParameters());
        return params;
    }
}