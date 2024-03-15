package RN.tests;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.algoactivations.EActivation;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.Node;

public class TestNodeWithBias {

	private Node node = null;
	private Link inputLink = null;
	private Link outputLink = null;
	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		node = new Node(EActivation.IDENTITY);
		node.createBias();
		inputLink = Link.getInstance(ELinkType.REGULAR, true);
		node.addInput(inputLink);
		outputLink = Link.getInstance(ELinkType.REGULAR, true);
		node.addOutput(outputLink);
	}

	@After
	public void tearDown() throws Exception {
		node = null;
	}

	@Test
	public void testDisconnect() {
		assertTrue(node.getInputs().size() == 1);
		assertTrue(node.getOutputs().size() == 1);
		node.disconnect();
		assertEquals(0, node.getInputs().size());
		assertEquals(0, node.getOutputs().size());
	}

	@Test
	public void testFinalizeConnections() {
		fail("Not yet implemented");
	}

	@Test
	public void testSublayerFanOutCrossLinkage() {
		fail("Not yet implemented");
	}

	@Test
	public void testNextLayerFanOutCrossLinkage() {
		fail("Not yet implemented");
	}

	@Test
	public void testLink() {
		fail("Not yet implemented");
	}

	@Test
	public void testDoubleLink() {
		fail("Not yet implemented");
	}

	@Test
	public void testSelfLink() throws Exception {
		node.selfLink();
		inputLink.setUnlinkedValue(2.0D);
		node.computeOutput(false);
		assertEquals(1.0D, node.getComputedOutput(), 0.0D);
		node.computeOutput(false);
		assertEquals(2.0D, node.getComputedOutput(), 0.0D);
		node.computeOutput(false);
		assertEquals(3.0D, node.getComputedOutput(), 0.0D);
		node.computeOutput(false);
		assertEquals(4.0D, node.getComputedOutput(), 0.0D);		
	}

	@Test
	public void testBias() {
		assertEquals(1D, node.getBiasInput().getValue(), 0D);
		assertEquals(1D, node.getBiasInput().getWeight(), 0D);
	}
	
	@Test
	public void testComputeOutput() throws Exception {
		
		node.computeOutput(false);
		assertEquals(-1.0D, node.getComputedOutput(), 0.0D);
		
		inputLink.setUnlinkedValue(2.0D);
		node.computeOutput(false);
		assertEquals(1.0D, node.getComputedOutput(), 0.0D);
		
		node.setActivationFx(EActivation.SYGMOID_0_1);
		inputLink.setUnlinkedValue(0.0D);
		node.computeOutput(false);
		assertEquals(0.2689D, node.getComputedOutput(), 0.00005D);
		
		inputLink.setUnlinkedValue(1.0D);
		node.computeOutput(false);
		assertEquals(0.5D, node.getComputedOutput(), 0.00005D);
		
		node.setActivationFx(EActivation.SYGMOID_1_1);
		inputLink.setUnlinkedValue(0.0D);
		node.computeOutput(false);
		assertEquals(-0.4621D, node.getComputedOutput(), 0.00005D);
		
		inputLink.setUnlinkedValue(1.0D);
		node.computeOutput(false);
		assertEquals(0D, node.getComputedOutput(), 0.00005D);
	}

	@Test
	public void testPerformActivationFunction() {
		fail("Not yet implemented");
	}

	@Test
	public void testPerformDerivativeFunction() {
		fail("Not yet implemented");
	}

}
