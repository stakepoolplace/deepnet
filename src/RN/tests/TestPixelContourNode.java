package RN.tests;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.Area;
import RN.AreaSquare;
import RN.ILayer;
import RN.Layer;
import RN.Network;
import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.ContourLinkage;
import RN.linkage.ELinkage;
import RN.linkage.FullFanOutLinkage;
import RN.linkage.OneToOneLinkage;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.Node;
import RN.nodes.PixelNode;

public class TestPixelContourNode {
	
	Network net = null;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		
		net = Network.newInstance(null);
		
		int nbPixels = 9;
		
		AreaSquare inArea = new AreaSquare(3, 3, false);
		//inArea.setLinkage(new OneToOneLinkage());
		inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, true).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
		Layer inLayer = new Layer(inArea);
		
		AreaSquare hiddenArea = new AreaSquare(3, 3, false);
		//hiddenArea.setLinkage(new ContourLinkage());
		hiddenArea.configureLinkage(ELinkage.SIMPLE_CONTOUR, null, true).configureNode(true, ENodeType.PIXEL).createNodes(nbPixels);
		ILayer hiddenLayer = new Layer(hiddenArea);
		
		AreaSquare outArea = new AreaSquare(1);
		//outArea.setLinkage(new FullFanOutLinkage());
		outArea.configureLinkage(ELinkage.MANY_TO_MANY, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(1);
		ILayer outLayer = new Layer(outArea);
		
		net.addLayer(inLayer, hiddenLayer, outLayer);
		
//		Node outputNode = new Node(EActivation.SYGMOID_0_1);
//		
//		outputNode.createBias();
//		
//		outArea.addNode(outputNode);
//		
//		
//		PixelNode hiddenNode = null;
//		PixelNode pixNode = null;
//
//		for(double idx = 0; idx < nbPixels; idx++){
//			
//			pixNode = new PixelNode();
//			
//			inArea.addNode(pixNode);
//			
//			pixNode.initParameters(nbPixels);
//
//			hiddenNode = new PixelNode();
//			hiddenNode.createBias();
//			hiddenArea.addNode(hiddenNode);
//			hiddenNode.initParameters(nbPixels);
//			
//			
//		}
		
		net.finalizeConnections();
		
		System.out.println(net.getString());
		
	}

	@After
	public void tearDown() throws Exception {
		net.initContext();
		net = null;
	}

	@Test
	public void testSublayerFanOutCrossLinkageLayer() {
		
		
		
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for(Link linkIn : net.getNode(1, 0, 0).getInputs()){
			map.put(linkIn.getSourceNode().getNodeId(), map.get(linkIn.getSourceNode().getNodeId()) == null ? 1 : (int) map.get(linkIn.getSourceNode().getNodeId()) + 1); 
		}
		
		System.out.println(net.getNode(1, 0, 0).getString());
		//System.out.println(net.getString());
		
		assertEquals(4, (int) map.get(0), 0);
		assertEquals(2, (int) map.get(1), 0);
		assertEquals(2, (int) map.get(3), 0);
		assertEquals(1, (int) map.get(4), 0);
		
		
	}



}
