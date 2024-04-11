package RN;

import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.Area;
import RN.Layer;
import RN.Network;
import RN.algoactivations.EActivation;
import RN.dataset.OutputData;
import RN.linkage.ELinkage;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;

public class TestNetworkClock {

	static Network net = null;
	static Layer layerIn = null;
	static Area area = null;
	static Layer layerHidden = null;
	static Layer layerOut = null;
	static Link lateralLink = null;
	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		
		net = Network.newInstance(null);
		
		layerIn = new Layer();
		net.addLayer(layerIn);
		area = new Area(2);
		layerIn.addArea(area);
		area.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.REGULAR).createNodes(2);
		
		
		layerHidden = new Layer();
		area = new Area(EActivation.SYGMOID_0_1, 2);
		layerHidden.addArea(area);
		net.addLayer(layerHidden);
		area.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode( true, EActivation.IDENTITY, ENodeType.REGULAR).createNodes(2);
		
		layerOut = new Layer();
		net.addLayer(layerOut);
		area = new Area(EActivation.SYGMOID_0_1, 1);
		layerOut.addArea(area);
		area.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode( true, EActivation.IDENTITY, ENodeType.REGULAR).createNodes(1);
		
		
		lateralLink = net.getNode(0, 0, 0).link(net.getNode(0, 0, 1), ELinkType.RECURRENT_LATERAL_LINK);
		
		net.finalizeConnections();
		
	}

	@After
	public void tearDown() throws Exception {
		net.initContext();
	}

	@Test
	public void testPropagation() throws Exception {
		OutputData output = net.propagation(false);
		//assertEquals(0.92144305166011564706487433388853755619310805569255489568539596908292519094472012246622682774908194908331567855012601684D, output.getOutput(0), 0);
		//assertEquals(new SygmoidPerformer().perform(new SygmoidPerformer().perform(-1) * 2 - 1), output.getOutput(0), 0);
		output = net.propagation(false);
		assertEquals(0.92144305166011564706487433388853755619310805569255489568539596908292519094472012246622682774908194908331567855012601684D, output.getOutput(0), 0);
		output = net.propagation(false);
		assertEquals(0.92144305166011564706487433388853755619310805569255489568539596908292519094472012246622682774908194908331567855012601684D, output.getOutput(0), 0);
		
	}
	
	@Test
	public void testClock() throws Exception {
		
		net.getNode(0, 0, 0).getInput(0).setUnlinkedValue(3.0D);
		
		assertEquals(-1, net.getContext().getClock());
		assertEquals(0, lateralLink.getFireTimeT());
		
		net.propagation(false);
		
		assertEquals(0, net.getContext().getClock());		
		assertEquals(1, lateralLink.getFireTimeT());
		
		net.propagation(false);
		
		assertEquals(1, net.getContext().getClock());		
		assertEquals(2, lateralLink.getFireTimeT());

	}
	

}
