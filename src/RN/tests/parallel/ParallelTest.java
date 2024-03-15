package RN.tests.parallel;

import java.util.concurrent.ForkJoinPool;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.AreaSquare;
import RN.ILayer;
import RN.Layer;
import RN.Network;
import RN.algoactivations.EActivation;
import RN.nodes.ImageNode;
import RN.nodes.PixelNode;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class ParallelTest extends Application{
	
	Network net = null;
	int nbPixels = 25;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		Thread t = new Thread("JavaFX Init Thread") {
	        public void run() {
	        	
	        	Application.launch(ParallelTest.class, new String[0]);
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
	public void testProcessors() {
		int processors = Runtime.getRuntime().availableProcessors();
		System.out.println("No of processors: " + processors);
		
		createMultiScaleNetwork();
		try {
			Thread.sleep(2000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		finalizeNet();

		PropagationProblem bigProblem = new PropagationProblem( 0, 1);
		bigProblem.layer = net.getLastLayer();

		Task task = new Task(bigProblem);
		ForkJoinPool pool = new ForkJoinPool(processors);
		pool.invoke(task);

	}
	
	private void finalizeNet(){
		
		net.finalizeConnections();
		
		//Image noir et blanc en input
		Image image = new Image("file:/Users/ericmarchand/Documents/workspace_neural/bikes.jpg");
		PixelReader pixelReader = image.getPixelReader();
		PixelNode pixNode = null;
		for(int idx = 0; idx < nbPixels; idx++){
			pixNode = (PixelNode) net.getFirstLayer().getArea(0).getNode(idx);
			
			Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
			pixNode.getInput(0).setUnlinkedValue(color.getBrightness());
			
		}
	}
	
	private void createMultiScaleNetwork(){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				
				
				ImageNode imageNode0 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode1 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode2 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode3 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode4 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode5 = new ImageNode(EActivation.IDENTITY);
				
				net = Network.getInstance();
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea_1_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea_1_0, hiddenArea_1_1, hiddenArea_1_2, hiddenArea_1_3, hiddenArea_1_4);
				
				AreaSquare hiddenArea2 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer1 = new Layer(hiddenArea2);

				AreaSquare hiddenArea_2_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer2 = new Layer(hiddenArea_2_0, hiddenArea_2_1, hiddenArea_2_2, hiddenArea_2_3, hiddenArea_2_4);
				
				AreaSquare hiddenArea3 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer3 = new Layer(hiddenArea3);
				
				AreaSquare hiddenArea_3_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer4 = new Layer(hiddenArea_3_0, hiddenArea_3_1, hiddenArea_3_2, hiddenArea_3_3, hiddenArea_3_4);

				AreaSquare outArea = new AreaSquare(nbPixels, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, hiddenLayer4, outLayer);
				
				
				
//				inArea.configureNode(false, EActivation.IDENTITY, ELinkage.ONE_TO_ONE, false, ENodeType.PIXEL).createNodes(nbPixels);
//				
//				double sigma0 = 0.3D;
//				double t = 0;
//				double k = Math.pow(2D, t) * 1.01D;
//				hiddenArea_1_0.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(sigma0, 1D).createNodes(nbPixels);
//				hiddenArea_1_1.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(k*sigma0, k).createNodes(nbPixels);
//				hiddenArea_1_2.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 2D)*sigma0, Math.pow(k, 2D)).createNodes(nbPixels);
//				hiddenArea_1_3.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 3D)*sigma0, Math.pow(k, 3D)).createNodes(nbPixels);
//				hiddenArea_1_4.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 4D)*sigma0, Math.pow(k, 4D)).createNodes(nbPixels);
//				
//				hiddenArea2.configureNode( false, EActivation.IDENTITY, ELinkage.ONE_TO_ONE_FETCH_AREA, false, ENodeType.PIXEL).createNodes(nbPixels);
//				
//				t = 1;
//				k = Math.pow(2D, t) * 1.01D;
//				hiddenArea_2_0.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 0)*sigma0, Math.pow(k, 0)).createNodes(nbPixels);
//				hiddenArea_2_1.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 1)*sigma0, Math.pow(k, 1)).createNodes(nbPixels);
//				hiddenArea_2_2.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 2)*sigma0, Math.pow(k, 2)).createNodes(nbPixels);
//				hiddenArea_2_3.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 3)*sigma0, Math.pow(k, 3)).createNodes(nbPixels);
//				hiddenArea_2_4.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 4)*sigma0, Math.pow(k, 4)).createNodes(nbPixels);
//				
//				hiddenArea3.configureNode( false, EActivation.IDENTITY, ELinkage.ONE_TO_ONE_FETCH_AREA, false, ENodeType.PIXEL).createNodes(nbPixels);
//				
//				t = 2;
//				k = Math.pow(2D, t) * 1.01D;
//				hiddenArea_3_0.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 0)*sigma0, Math.pow(k, 0)).createNodes(nbPixels);
//				hiddenArea_3_1.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 1)*sigma0, Math.pow(k, 1)).createNodes(nbPixels);
//				hiddenArea_3_2.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 2)*sigma0, Math.pow(k, 2)).createNodes(nbPixels);
//				hiddenArea_3_3.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 3)*sigma0, Math.pow(k, 3)).createNodes(nbPixels);
//				hiddenArea_3_4.configureNode(false, EActivation.IDENTITY, ELinkage.DOG, false, ENodeType.GAUSSIAN).configureNodeOptParams(Math.pow(k, 4)*sigma0, Math.pow(k, 4)).createNodes(nbPixels);
//				
//				outArea.configureNode( false, EActivation.IDENTITY, ELinkage.ONE_TO_ONE_FETCH_AREA, false, ENodeType.REGULAR).createNodes(nbPixels);
				
				
//				System.out.println("Begin propagation...");
//				try {
//					net.propagation(false);
//				} catch (Exception e) {
//					e.printStackTrace();
//				}
//				System.out.println("End propagation.");
				
				
				//System.out.println(net.getNode(3, 0, 0).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(2, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_V1Orientation);
		
			}});

	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
