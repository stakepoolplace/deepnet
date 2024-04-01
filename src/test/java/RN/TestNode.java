package RN;

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

public class TestNode {

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
		assertTrue(node.getInputs().size() > 0);
		assertTrue(node.getOutputs().size() > 0);
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
		inputLink.setUnlinkedValue(1.0D);
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
	public void testComputeOutput() throws Exception {
		
		node.computeOutput(false);
		assertEquals(0.0D, node.getComputedOutput(), 0.0D);
		
		inputLink.setUnlinkedValue(2.0D);
		node.computeOutput(false);
		assertEquals(2.0D, node.getComputedOutput(), 0.0D);
		
		node.setActivationFx(EActivation.SYGMOID_0_1);
		inputLink.setUnlinkedValue(0.0D);
		node.computeOutput(false);
		assertEquals(0.5D, node.getComputedOutput(), 0.0D);
		
		node.setActivationFx(EActivation.SYGMOID_1_1);
		inputLink.setUnlinkedValue(0.0D);
		node.computeOutput(false);
		assertEquals(0.0D, node.getComputedOutput(), 0.0D);
	}

	@Test
	public void testPerformActivationFunction() {
		fail("Not yet implemented");
	}

	@Test
	public void testPerformDerivativeFunction() {
		
//		double d = 1234567890.123456;
//		int i = new Double(d).intValue(); //recuperer la partie entiere
//		double decimale = d-(new Double(i).doubleValue());
		
		int nbWidth = 10000;
		
		for(int n = 0; n < nbWidth; n+=100){
			int width = n;
			int x;
			int y;
			Double val;
			int index = 0;
			System.out.println("width="+n);
			for(int idy = 0; idy < width; idy++){
				for(int idx = 0; idx < width; idx++){
					
					val = new Double(index / width);
					y = val.intValue();
					x =  index - y * width;
					
					//System.out.println("reel (x,y) = ("+ idx+","+idy+")  ->   val=" + val + "  (x,y) = ("+x+","+y+")");
					
					assertEquals(x, idx, 0.0D);
					assertEquals(y, idy, 0.0D);
					
					index++;
				}
			}
		
		}
		
		
	}
	

}
