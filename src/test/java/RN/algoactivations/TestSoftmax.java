package RN.algoactivations;

import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import RN.ENetworkImplementation;
import RN.ITester;
import RN.Layer;
import RN.Network;
import RN.TestNetwork;
import RN.algotrainings.BackPropagationWithCrossEntropyTrainer;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.InputSample;
import RN.linkage.ELinkage;
import RN.nodes.ENodeType;

/**
 * @author Eric Marchand
 *
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class TestSoftmax {
	
	
	private static Network network = null;
	private static ITester tester = null;
	private static ITrainer trainer = null;
	private static int nbIterations;
	
	@BeforeClass
	public static void setUp() {
	
		// Initialisation du réseau
		network = Network.getInstance(ENetworkImplementation.LINKED).setName("SimpleFFNetwork");

		// Configuration des couches
		int inputSize = 2; // Taille de l'entrée
		int hiddenSize = 3; // Nombre de neurones dans la couche cachée
		int outputSize = 2; // Taille de la sortie
		network.addLayer(new Layer("InputLayer", inputSize));
		network.addLayer(new Layer("HiddenLayer1", hiddenSize));
		network.addLayer(new Layer("OutputLayer", EActivation.SOFTMAX, outputSize));

		// Création des connexions entre les couches
		// Note: Implémentez la logique de connexion dans vos classes Layer/Node
		network.getFirstLayer().getArea(0).configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false,
		    EActivation.IDENTITY, ENodeType.REGULAR).createNodes(inputSize);

		// ELU et SYGMOID fonctionnent bien
		network.getLayer(1).getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true).configureNode(true,
		    EActivation.ELU, ENodeType.REGULAR).createNodes(hiddenSize);

		// no bias for softmax
		network.getLastLayer().getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true)
		    .configureNode(false, ENodeType.REGULAR).createNodes(outputSize);			

		network.finalizeConnections();
		
		// Initialisation de l'entraîneur et configuration des paramètres d'entraînement
		tester = TestNetwork.getInstance();
		trainer = new BackPropagationWithCrossEntropyTrainer();
		tester.setNetwork(network);
		trainer.setLearningRate(0.1);
        trainer.setMomentum(0.9);
        
        nbIterations = 100;
		
		
	}
	
	@Test
	public void testA_Train() throws Exception {
		
		
		// Chargement des données d'entraînement
		// Note: Adaptez cette partie pour charger vos propres données
		loadData("./src/test/resources/datasets/dataset-SOFTMAX.csv");
		
		// Initialisation aléatoire des poids du réseau
		tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
		
		// Entraînement du réseau
		trainer.launchTrain(nbIterations);	
		
	}
	
	
	@Test
	public void testB_Perform() throws Exception {
		
		network.getNode(0, 0, 0).getInput(0).setUnlinkedValue(0.1D);
		network.getNode(0, 0, 1).getInput(0).setUnlinkedValue(0.2D);
		
		network.getNode(2, 0, 0).setIdealOutput(1D);
		network.getNode(2, 0, 1).setIdealOutput(0D);

		network.init(0, 1);
		

		//network.show();
		
		network.propagation(false);
		trainer.backPropagateError();

		
		//network.show();
		
		double errorBeforeTrain = trainer.getErrorRate();
		network.propagation(false);
		trainer.backPropagateError();
		network.propagation(false);
		trainer.backPropagateError();
		double errorAfterTrain = trainer.getErrorRate();

		double sum = network.getNode(2, 0, 0).getComputedOutput() + network.getNode(2, 0, 1).getComputedOutput();

		assertEquals(1D, sum, 0.01);
		assertTrue(errorBeforeTrain > errorAfterTrain);
	}

	
	
	private static void loadData(String filePath) {

		if (filePath != null) {

			try {
				File file = new File(filePath); 
				InputSample.setCSVFileDataset(file.getPath());
				System.out.println("Dataset chargé.");
			} catch (IOException ioe) {
				ioe.printStackTrace();
			}

		}

	}

}
