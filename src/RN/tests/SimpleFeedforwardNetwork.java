package RN.tests;

import java.io.File;
import java.io.IOException;

import RN.ITester;
import RN.Layer;
import RN.Network;
import RN.TestNetwork;
import RN.algoactivations.EActivation;
import RN.algotrainings.BackPropagationTrainer;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.InputSample;
import RN.linkage.ELinkage;
import RN.nodes.ENodeType;

public class SimpleFeedforwardNetwork {

	public static void main(String[] args) {
		try {
			// Initialisation du réseau
			Network network = Network.getInstance().setName("SimpleFFNetwork");

			// Configuration des couches
			int inputSize = 2; // Taille de l'entrée
			int hiddenSize = 2; // Nombre de neurones dans la couche cachée
			int outputSize = 1; // Taille de la sortie
			network.addLayer(new Layer("InputLayer", inputSize));
			network.addLayer(new Layer("HiddenLayer1", hiddenSize));
			network.addLayer(new Layer("OutputLayer", outputSize));

			// Création des connexions entre les couches
			// Note: Implémentez la logique de connexion dans vos classes Layer/Node
			network.getFirstLayer().getArea(0).configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false,
					EActivation.IDENTITY, ENodeType.REGULAR).createNodes(inputSize);
			
			network.getLayer(1).getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true).configureNode(true,
					EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(hiddenSize);
			
			network.getLastLayer().getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true)
					.configureNode(true, EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(outputSize);			
			
			network.finalizeConnections();
			
			
			// Initialisation de l'entraîneur et configuration des paramètres d'entraînement
			ITester tester = TestNetwork.getInstance();
			ITrainer trainer = new BackPropagationTrainer();
			tester.setNetwork(network);
			trainer.setLearningRate(0.1);
            trainer.setMomentum(0.9);

			// Chargement des données d'entraînement
			// Note: Adaptez cette partie pour charger vos propres données
			loadData("./data/dataset-OR.csv");
			
			// Initialisation aléatoire des poids du réseau
			tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
			

			System.out.println(network.getString());

			// Entraînement du réseau
			trainer.launchTrain(20000); // 20 000 itérations

			System.out.println("Entraînement terminé.");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}



	private static void loadData(String filePath) {

		if (filePath != null) {

			try {
				File file = new File(filePath); // Remplacez ceci par le chemin de votre dossier
				InputSample.setCSVFileDataset(file.getPath());
				System.out.println("Dataset chargé.");
			} catch (IOException ioe) {
				ioe.printStackTrace();
			}

		}

	}
}
