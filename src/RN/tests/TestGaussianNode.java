package RN.tests;

import static org.junit.Assert.fail;

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
import RN.nodes.Node;
import RN.nodes.PixelNode;

public class TestGaussianNode {

	PixelNode gaussNode = null;
	int nbPixels = 10000;
	Network net = Network.getInstance();
	AreaSquare inArea = new AreaSquare(nbPixels);
	Layer inLayer = new Layer(inArea);
	Area hiddenArea = new Area(1);
	ILayer hiddenLayer = new Layer(hiddenArea);
	Area outArea = new Area(1);
	ILayer outLayer = new Layer(outArea);
	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		
		gaussNode = new PixelNode();
		
		net.addLayer(inLayer, hiddenLayer, outLayer);
		
		Node outputNode = new Node(EActivation.SYGMOID_0_1);
		outputNode.createBias();
		//outputNode.setInnerNode(imageNode);
		
		outArea.addNode(outputNode);
		
		//imageNode.initImageScene(outputNode);
		
		PixelNode pixNode = null;
		for(double idx = 0; idx < nbPixels; idx++){
			
			pixNode = new PixelNode();
			inArea.addNode(pixNode);
			
			pixNode.initParameters(nbPixels);
			
		}

	}

	@After
	public void tearDown() throws Exception {
		gaussNode = null;
		net = null;
	}

	@Test
	public void testSublayerFanOutCrossLinkageLayer() {
		
		net.finalizeConnections();
		
		//System.out.println(gaussNode.getString());
		
		double margeCarre = 30D;
		for(int idx = 0; idx < nbPixels; idx++){
			PixelNode pixNode = (PixelNode) inArea.getNode(idx);
			// carré
			if(pixNode.getX() > margeCarre && pixNode.getX() < inArea.getWidthPx() - margeCarre && pixNode.getY() > margeCarre && pixNode.getY() < inArea.getHeightPx() - margeCarre)
				pixNode.getInput(0).setUnlinkedValue(1D);
			else
				pixNode.getInput(0).setUnlinkedValue(0D);
			
		}
	}

	@Test
	public void testComputeOutput() {

		//System.out.println(net.getNode(1, 0, 3031).getString());
		
		try {
			net.propagation(false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Test
	public void testGetFilterValue() {
		fail("Not yet implemented");
	}

}
