package RN.tests;

import static org.junit.Assert.fail;
import junit.framework.Assert;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.algoactivations.EActivation;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.LSTMNode;
import RN.nodes.Node;

public class TestLSTMNode {

	private LSTMNode node = null;
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
		node = new LSTMNode(1);
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
//		Assert.assertTrue(node.getInputs().size() > 0);
//		Assert.assertTrue(node.getOutputs().size() > 0);
//		node.disconnect();
//		Assert.assertEquals(0, node.getInputs().size());
//		Assert.assertEquals(0, node.getOutputs().size());
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
//		node.selfLink();
//		inputLink.setUnlinkedValue(1.0D);
//		node.computeOutput(false);
//		Assert.assertEquals(1.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		Assert.assertEquals(2.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		Assert.assertEquals(3.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		Assert.assertEquals(4.0D, node.getComputedOutput(), 0.0D);		
	}

	@Test
	public void testComputeOutput() throws Exception {
		
		node.computeOutput(false);
		Assert.assertEquals(0.0D, node.getComputedOutput(), 0.0D);
		
		inputLink.setUnlinkedValue(2.0D);
		node.computeOutput(false);
		Assert.assertEquals(2.0D, node.getComputedOutput(), 0.0D);
		
		node.setActivationFx(EActivation.SYGMOID_0_1);
		inputLink.setUnlinkedValue(0.0D);
		node.computeOutput(false);
		Assert.assertEquals(0.5D, node.getComputedOutput(), 0.0D);
	}

	@Test
	public void testPerformActivationFunction() {
		fail("Not yet implemented");
	}

	@Test
	public void testPerformDerivativeFunction() {
		fail("Not yet implemented");
	}
	


	@Test
	public void testGetString() {
		fail("Not yet implemented");
	}

	@Test
	public void testLSTMNode() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetInput() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetInputGate() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetForgetGate() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetOutputGate() {
		fail("Not yet implemented");
	}

	@Test
	public void testIsBidirectional() {
		fail("Not yet implemented");
	}

	@Test
	public void testSetBidirectional() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetMemory() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetOutputProductUnit() {
		fail("Not yet implemented");
	}

}
