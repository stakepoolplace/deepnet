package RN.algoactivations;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import RN.Layer;
import RN.Network;
import RN.linkage.ELinkage;
import RN.nodes.ENodeType;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class TestSoftmax {
	
	
	private static Network network = null;
	
	@Before
	public void setUp() {
	
		// Initialisation du réseau
		network = Network.getInstance().setName("SimpleFFNetwork");

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

		// no bias for softmax
		network.getLayer(1).getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode(false,
		    EActivation.SOFTMAX, ENodeType.REGULAR).createNodes(hiddenSize);

		network.getLastLayer().getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true)
		    .configureNode(true, EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(outputSize);			

		network.finalizeConnections();
		
		
		
		
	}
	
	
	@Test
	public void testPerform() throws Exception {
		
		network.getNode(0, 0, 0).getInput(0).setUnlinkedValue(3.0D);
		network.getNode(0, 0, 1).getInput(0).setUnlinkedValue(2.0D);

		
		//network.show();
		network.init(0, 1);
		
		network.propagation(false);

		network.show();

		assertEquals(0.73D, network.getNode(1, 0, 0).getComputedOutput(), 0.01);
		assertEquals(0.26D, network.getNode(1, 0, 1).getComputedOutput(), 0.01);
	}

	@Test
	public void testPerformDerivative() throws Exception {
		
		INode node = network.getNode(1, 0, 0);
		SoftMaxPerformer syg = new SoftMaxPerformer(node);
		assertEquals(-56D, syg.performDerivative(8D, 0.5D, 4D, 9D), 0.01);
	}

}
