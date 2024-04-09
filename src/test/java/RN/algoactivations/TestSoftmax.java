package RN.algoactivations;

import static org.junit.Assert.assertEquals;

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
	
	@BeforeClass
	public static void setUp() {
	
		// Initialisation du réseau
		network = Network.getInstance(ENetworkImplementation.LINKED).setName("SimpleFFNetwork");

		// Configuration des couches
		int inputSize = 2; // Taille de l'entrée
		int hiddenSize = 2; // Nombre de neurones dans la couche cachée
		int outputSize = 2; // Taille de la sortie
		network.addLayer(new Layer("InputLayer", inputSize));
		network.addLayer(new Layer("HiddenLayer1", hiddenSize));
		network.addLayer(new Layer("OutputLayer", EActivation.SOFTMAX, outputSize));

		// Création des connexions entre les couches
		// Note: Implémentez la logique de connexion dans vos classes Layer/Node
		network.getFirstLayer().getArea(0).configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false,
		    EActivation.IDENTITY, ENodeType.REGULAR).createNodes(inputSize);

		// no bias for softmax
		network.getLayer(1).getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true).configureNode(true,
		    EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(hiddenSize);

		network.getLastLayer().getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true)
		    .configureNode(false, EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(outputSize);			

		network.finalizeConnections();
		
		// Initialisation de l'entraîneur et configuration des paramètres d'entraînement
		tester = TestNetwork.getInstance();
		trainer = new BackPropagationWithCrossEntropyTrainer();
		tester.setNetwork(network);
		trainer.setLearningRate(0.1);
        trainer.setMomentum(0.9);
		
		
	}
	
	@Test
	public void testA_Train() throws Exception {
		
		


		// Chargement des données d'entraînement
		// Note: Adaptez cette partie pour charger vos propres données
		loadData("./src/test/resources/datasets/dataset-SOFTMAX.csv");
		
		// Initialisation aléatoire des poids du réseau
		tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
		

		System.out.println(network.getString());

		// Entraînement du réseau
		trainer.launchTrain(20000); // 20 000 itérations	
		
		System.out.println(network.getString());
		
	}
	
	
	@Test
	public void testB_Perform() throws Exception {
		
		network.getNode(0, 0, 0).getInput(0).setUnlinkedValue(3.0D);
		network.getNode(0, 0, 1).getInput(0).setUnlinkedValue(2.0D);

		network.init(0, 1);
		
		network.show();

		trainer.setCurrentOutputData(network.propagation(false));
		
		network.show();

		trainer.backPropagateError();
		
		network.show();

		assertEquals(0.73D, network.getNode(1, 0, 0).getComputedOutput(), 0.01);
		assertEquals(0.26D, network.getNode(1, 0, 1).getComputedOutput(), 0.01);
	}
//	
//	
//
//	@Test
//	public void testC_PerformDerivative() throws Exception {
//		
//		INode node = network.getNode(1, 0, 0);
//		SoftMaxPerformer syg = new SoftMaxPerformer(node.getArea());
//		assertEquals(-56D, syg.performDerivative(8D, 0.5D, 4D, 9D), 0.01);
//	}
	

	
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
