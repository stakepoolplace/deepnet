package RN.tests;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.Area;
import RN.AreaSquare;
import RN.ENetworkImplementation;
import RN.ILayer;
import RN.Layer;
import RN.Network;
import RN.algoactivations.EActivation;
import RN.linkage.ELinkage;
import RN.linkage.OneToOneFetchOctaveAreaLinkage;
import RN.linkage.vision.KeyPoint;
import RN.nodes.ENodeType;
import RN.nodes.PixelNode;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class HessianCourbureTest extends Application{

	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		
		Thread t = new Thread("JavaFX Init Thread") {
	        public void run() {
	        	
	        	Application.launch(TestImageNode.class, new String[0]);
	        }
	    };
	    t.setDaemon(true);
	    t.start();
	    Thread.sleep(1000);
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testCourbure()  {
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				int nbPixels = 10000;
				int pixelGaussianNodes = 10000;
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
//				Area hidden2Area = new Area(nbPixels, true);
//				Layer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(1, true);
				ILayer outLayer = new Layer(outArea);
				net.addLayer(inLayer, hiddenLayer, outLayer);
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false);
				inArea.configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				
				double k = 1.6D;
				//double k = 1.106D;
				double f = 2D;
				hiddenArea.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 5D), f * 1D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);
				
				//hidden2Area.configureLinkage(ELinkage.HESSIAN_COURBURE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.GAUSSIAN).createNodes(pixelGaussianNodes);
				
				outArea.configureLinkage(ELinkage.ONE_TO_ONE_FETCH_OCTAVE_AREA, null, false).configureNode( false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);

				
				
				net.finalizeConnections();
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Image image = new Image("file:/Users/ericmarchand/Documents/workspace_neural/clochers.jpg");
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getBrightness());
					
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				for(KeyPoint point : ((OneToOneFetchOctaveAreaLinkage)outArea.getLinkage()).getKeyPoints()){
					System.out.println(point);
				}
				
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});
		
		try {
			Thread.sleep(500000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	}
	
	@Test
	public void testArcTan2(){
	
		for(int idy = -3; idy <= 3; idy++){
			for(int idx = -3; idx <= 3; idx++){
				System.out.println("x=" + idx + " y=" + idy + " theta=" + (Math.atan2(idy, idx) * 180D) / Math.PI);
			}
		}
		
	}
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		// TODO Auto-generated method stub
		
	}
}
