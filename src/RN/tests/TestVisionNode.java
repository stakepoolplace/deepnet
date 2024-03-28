package RN.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.List;

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
import RN.linkage.ELinkageBetweenAreas;
import RN.linkage.FilterLinkage;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.ImageNode;
import RN.nodes.PixelNode;
import RN.utils.MathUtils;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class TestVisionNode extends Application{

	private ImageNode imageNode = null;
	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		
		Thread t = new Thread("JavaFX Init Thread") {
	        public void run() {
	        	
	        	Application.launch(TestVisionNode.class, new String[0]);
	        }
	    };
	    t.setDaemon(true);
	    t.start();
	    Thread.sleep(1000);
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
		System.out.println("Fin des tests....");
		Thread.sleep(100000);
	}

	@Before
	public void setUp() throws Exception {
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				imageNode = new ImageNode(EActivation.IDENTITY);
				
			}
			
		});

	}

	@After
	public void tearDown() throws Exception {
		
		imageNode = null;
		System.out.println("----- Next test -----");
	}


	

	
	@Test
	public void testShowCenterPointImage() {
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				int nbPixels = 10000;
				AreaSquare area = new AreaSquare(nbPixels);
				PixelNode centerPix = new PixelNode();
				Link in = Link.getInstance(ELinkType.REGULAR, true);
				in.setUnlinkedValue(1D);
				centerPix.addInput(in);
				area.addNode(centerPix);
				centerPix.setNodeId(5050);
				centerPix.initParameters();
				
				concentricCircleCenters(10D, 11D, area);
				

				assertEquals(50D, centerPix.getX(), 0.0D);
				assertEquals(50D, centerPix.getY(), 0.0D);
				assertEquals(100.0, area.getWidthPx(), 0.0D);
				assertEquals(area.getWidthPx(), area.getHeightPx(), 0.0D);
				
				AreaSquare area1 = new AreaSquare(1);
				PixelNode hiddenPix = new PixelNode();
				area1.addNode(hiddenPix);
				
				
				try {
					
					for(INode pix : area.getNodes()){
						pix.link(hiddenPix, ELinkType.REGULAR);
						pix.computeOutput(false);
					}
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				imageNode.showImage(hiddenPix);
				
			}
		});
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	}
	
	private void concentricCircleCenters(Double angleCount, Double scaleCount, Area area) {
		
		double base = 1 + (Math.PI / (Math.sqrt(3D) * angleCount));
		double arc = 2D * Math.PI / angleCount;
		
		double p_fovea = angleCount / (2D * Math.PI) + 2;
		
		Double p_r = null;
		
		//double t_radius = arc * 1/3;
		
		int idCircle = 0;
		double theta = 0D;
		
		for(int idScale=0; idScale <= scaleCount; idScale++){
			for(int idAngle=0; idAngle <= angleCount; idAngle++){
				
				theta = arc * (idAngle + MathUtils.odd(idScale) / 2D);
				p_r = p_fovea * Math.pow(base, idScale);
				
				theta = MathUtils.round(theta, 2);
				
				createPix(area, theta, p_r, base);
				
				idCircle++;
				
			}
		}
	}
	
	void createPix(Area area, Double theta, Double p_r, Double base){
		
		PixelNode pix = new PixelNode();
		Link in = Link.getInstance(ELinkType.REGULAR, true);
		in.setUnlinkedValue(1D);
		pix.addInput(in);
		area.addNode(pix);
		
		pix.setTheta(theta);
		pix.setP(p_r);
		pix.getCoordinate().setBase(base);
		pix.getCoordinate().setX0((double) pix.getAreaSquare().getNodeCenterX());
		pix.getCoordinate().setY0((double) pix.getAreaSquare().getNodeCenterY());
		
		pix.getCoordinate().logPolarToLinearSystem();
		
		pix.setNodeId(Math.max(0, pix.getAreaSquare().nodeXYToNodeId(pix.getCoordinate().getX().intValue(), pix.getCoordinate().getY().intValue())));
		pix.initParameters();
		
	}
	
	@Test
	public void testShowSimpleImage() {
		showImageSimple();
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	
	private void showImageSimple(){
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				AreaSquare areaOut = new AreaSquare(1);
				PixelNode outNode = new PixelNode();
				areaOut.addNode(outNode);
				//outNode.createBias();
				outNode.setActivationFx(EActivation.SYGMOID_0_1);
				outNode.setInnerNode(imageNode);
				
				int nbPixels = 10000;
				AreaSquare areaIn = new AreaSquare(nbPixels);
				PixelNode pixNode = null;
				Link link = null;
				double margeCarre = 5D;
				
				for(double idx = 1; idx <= nbPixels; idx++){
					
					pixNode = new PixelNode();
					areaIn.addNode(pixNode);
					pixNode.setNodeId((int) idx - 1);
					pixNode.initParameters(nbPixels);
					pixNode.link(outNode, ELinkType.REGULAR, false);
					link = Link.getInstance(ELinkType.REGULAR, false);
					pixNode.addInput(link);
					
					// les photorécepteurs sont continuellements activés même en l'absence de stimulis -> 0.5D
					if(pixNode.getX() > margeCarre && pixNode.getX() < (pixNode.getAreaSquare().getWidthPx() - margeCarre) && pixNode.getY() > margeCarre && pixNode.getY() < (pixNode.getAreaSquare().getHeightPx() - margeCarre))
						link.setUnlinkedValue(1D);
					else
						link.setUnlinkedValue(0D);
					
					
					try {
						pixNode.computeOutput(false);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}

				imageNode.showImage(outNode);
		
			}});
		

	}
	
	

	
	@Test
	public void testLearningConvolImage() {
		
		
//		Le traitement repose sur cinq paramètres :
//		
//			N représente la taille du masque (matrice carrée) implantant le filtre LOG. N est impair.
//			σ permet d'ajuster la taille du chapeau mexicain.
//			∆x et ∆y sont les pas d'échantillonnage utilisés pour discrétiser h''(x,y). Généralement ∆x = ∆ y
//			S est le seuil qui permet de sélectionner les contours les plus marqués.
//		
//			Il est à noter que le choix des paramètres N, σ et ∆x ne doit pas se faire de façon indépendante. 
//  		En effet, le masque, même de taille réduite, doit ressembler à un chapeau mexicain. Le problème ici est le même que celui que l'on rencontre lors de l'échantillonnage d'une fonction gaussienne. 
//          Le nombre de points N à considérer doit être tel que l'étendue occupe l'intervalle [-3σ , 3σ].
//			En fonction du pas d'échantillonnage, l'étendue spatiale vaut : (N-1) ∆x  .
//			Cette étendue peut aussi s'écrire en fonction de σ : (N-1) ∆x = kσ  avec k entier.
//			En prenant par exemple  ∆x = 1 , il s'agit de choisir N et σ de sorte que l'étendue du chapeau mexicain soit pertinente. 
//  		Pour le chapeau mexicain, la valeur de k doit être au moins de 4.
		
		double N = 3D;
		double deltaX = 1;
		double k = 4D;
		double Ox = (N-1) * deltaX / k;
		double Oy = Ox;
		double alpha = 1D;
		double Mu = 0D;
		
		//showSimpleLogGaborVision(Ox, Oy, Mu, alpha);
		//showLogGaborVision(Ox, Oy, Mu, alpha);
		//showLogPolarRetina();
		showVision(Ox, Oy, Mu, alpha);
		//showVisionOnOff(Ox, Oy, Mu, alpha);
		
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}
	

	
	private void showVision( final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				Image image = new Image("file:./resources/a-16x16.png");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				
				int sampling = 1;
				int nbPixels = width * height;
				System.out.println("nb pixels inputs :" + nbPixels);
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
				AreaSquare hidden2Area = new AreaSquare(nbPixels, true);
				AreaSquare hidden2Area2 = new AreaSquare(nbPixels, true);
				AreaSquare hidden2Area3 = new AreaSquare(nbPixels, true);
				AreaSquare hidden2Area4 = new AreaSquare(nbPixels, true);
				ILayer hidden2Layer = new Layer(hidden2Area, hidden2Area2, hidden2Area3, hidden2Area4);
				
				AreaSquare hidden3Area = new AreaSquare(nbPixels, true);
				ILayer hidden3Layer = new Layer(hidden3Area);
				
				AreaSquare outArea = new AreaSquare(1);
				ILayer outLayer = new Layer(outArea);
				net.addLayer(inLayer, hiddenLayer, hidden2Layer, hidden3Layer, outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				hiddenArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false, 3D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				
				hidden2Area.configureLinkage(ELinkage.V1_ORIENTATIONS, ELinkageBetweenAreas.ONE_TO_MANY, null, false,  0D, 0.5D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hidden2Area.getImageArea().scaleImage(2);
				
				hidden2Area2.configureLinkage(ELinkage.V1_ORIENTATIONS, ELinkageBetweenAreas.ONE_TO_MANY, null, false,  45D, 0.5D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hidden2Area2.getImageArea().scaleImage(2);
				
				hidden2Area3.configureLinkage(ELinkage.V1_ORIENTATIONS, ELinkageBetweenAreas.ONE_TO_MANY, null, false,  90D, 0.5D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hidden2Area3.getImageArea().scaleImage(2);
				
				hidden2Area4.configureLinkage(ELinkage.V1_ORIENTATIONS, ELinkageBetweenAreas.ONE_TO_MANY, null, false,  135D, 0.5D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hidden2Area4.getImageArea().scaleImage(2);
				
				
				
				hidden3Area.configureLinkage(ELinkage.MAX, ELinkageBetweenAreas.MANY_TO_ONE, null, false, 3D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hidden3Area.getImageArea().scaleImage(2);
				
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.ONE_TO_ONE, null, true).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				
				
				net.finalizeConnections();
				
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Color color = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int y = 0; y < height; y+=sampling){
					for(int x = 0; x < width; x+=sampling){
							color = pixelReader.getColor(x, y);
							pixNode = (PixelNode) inArea.getNodeXY(x, y, sampling);
							pixNode.setEntry(color.getOpacity());
					}
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				hidden2Area.getFilter(FilterLinkage.ID_FILTER_V1Orientation).filterToImage(8);
				hidden2Area2.getFilter(FilterLinkage.ID_FILTER_V1Orientation).filterToImage(8);
				hidden2Area3.getFilter(FilterLinkage.ID_FILTER_V1Orientation).filterToImage(8);
				hidden2Area4.getFilter(FilterLinkage.ID_FILTER_V1Orientation).filterToImage(8);
				
				//System.out.println(net.getNode(2, 0, 48).getString());
				//net.getNode(1, 0, 5050).getArea().getFilter().filterToString(0);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	
	@Test
	public void showVisionOnOff(){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				int nbPixels = 10000;
				
				ImageNode imageNodeON = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNodeOFF = new ImageNode(EActivation.IDENTITY);
				
				ImageNode imageNodeBipL = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNodeBipS = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode0 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNodeGangOFF = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNodeGangON = new ImageNode(EActivation.IDENTITY);
				
				Network net = Network.newInstance(ENetworkImplementation.UNLINKED);
				
				Area inArea = new Area(nbPixels);
				Layer inLayer = new Layer(inArea);
				
				Area hiddenArea0 = new Area(nbPixels);
				Area hiddenArea1 = new Area(nbPixels);
				ILayer hiddenLayer = new Layer(hiddenArea0, hiddenArea1);
				
				Area hidden1Area0 = new Area(nbPixels);
				Area hidden1Area1 = new Area(nbPixels);
				ILayer hidden1Layer = new Layer(hidden1Area0, hidden1Area1);
				
				Area outArea0 = new Area(1);
				Area outArea1 = new Area(1);
				ILayer outLayer = new Layer(outArea0, outArea1);
				
				net.addLayer(inLayer, hiddenLayer, hidden1Layer, outLayer);
				
				
//				outArea0.configureNode( true, EActivation.SYGMOID_0_1, ELinkage.MANY_TO_MANY, false, ENodeType.REGULAR).createNodes(1);
//				outArea1.configureNode( true, EActivation.SYGMOID_0_1, ELinkage.MANY_TO_MANY, false, ENodeType.REGULAR).createNodes(1);
				
				PixelNode pixNode = null;
				
				System.out.println("Begin creation bipolars and ganglionaries nodes...");
				
				int pixelGanglionaryNodes = 10000;
				
//				hiddenArea0.configureNode( false, ELinkage.BIPOLAR, false, ENodeType.BIPOLAR_S).createNodes(pixelGanglionaryNodes);
//				hiddenArea1.configureNode( false, ELinkage.BIPOLAR, false, ENodeType.BIPOLAR_L).createNodes(pixelGanglionaryNodes);
//				
//				hidden1Area0.configureNode( false, ELinkage.GANGLIONARY, false, ENodeType.GANGLIONARY_OFF).createNodes(pixelGanglionaryNodes);
//				hidden1Area1.configureNode( false, ELinkage.GANGLIONARY, false, ENodeType.GANGLIONARY_ON).createNodes(pixelGanglionaryNodes);
//				
//				System.out.println("End creation bipolars and ganglionaries nodes.");
//				
//				net.getNode(1, 0, 5050).setInnerNode(imageNodeBipL);
//				net.getNode(1, 1, 5050).setInnerNode(imageNodeBipS);
//				
//				net.getNode(2, 0, 5050).setInnerNode(imageNodeGangON);
//				net.getNode(2, 1, 5050).setInnerNode(imageNodeGangOFF);
//
//				System.out.println("Begin creation input nodes...");
//				inArea.configureNode( false, ELinkage.ONE_TO_ONE, false, ENodeType.PIXEL).createNodes(pixelGanglionaryNodes);
				
				net.getNode(0, 0, 5050).setInnerNode(imageNode0);
				
				net.getNode(3, 0, 0).setInnerNode(imageNodeON);
				net.getNode(3, 1, 0).setInnerNode(imageNodeOFF);
				
				System.out.println("End creation input nodes.");
				
				net.finalizeConnections();
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				Image image = new Image("file:./resources/logo-transparent.png");
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.getInput(0).setUnlinkedValue(color.getBrightness());
					
				}
				System.out.println("End lecture image.");
				
				
				System.out.println("Begin propagation...");
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				System.out.println("End propagation.");
				
				System.out.println(net.getNode(2, 0, 5050).getString());
				System.out.println(net.getNode(2, 1, 5050).getString());
		
			}});

	}
	
	@Test
	public void testSimpleLogGaborVision(){
		
		showSimpleLogGaborVision();
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void showSimpleLogGaborVision(){
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				//Image image = new Image("file:./resources/empreinte.png");
				Image image = new Image("file:./resources/a-16x16.png");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				
				int sampling = 1;
				int nbPixels = (int) ((width / sampling) * (height / sampling)) ;
				System.out.println("nb pixels inputs :" + nbPixels);
				
				int samplingH = 1;
				int nbPixelsHidden = (int) ((width / samplingH) * (height / samplingH)) ;
				System.out.println("nb pixels hidden :" + nbPixelsHidden);
				
				int samplingH2 = 1;
				int nbPixelsHidden2 = 25;//(int) (nbPixelsHidden / Math.pow(samplingH2, 2));
				System.out.println("nb pixels hidden2 :" + nbPixelsHidden2);
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(width, height, true);
				AreaSquare hiddenArea2 = new AreaSquare(width, height, true);
				AreaSquare hiddenArea3 = new AreaSquare(width, height, true);

				ILayer hiddenLayer = new Layer(hiddenArea, hiddenArea2, hiddenArea3);
				
				AreaSquare hidden2Area = new AreaSquare(nbPixelsHidden2, true);
				ILayer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(10, 1, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, hidden2Layer, outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				hiddenArea.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 1D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea.getImageArea().scaleImage(2);
				hiddenArea2.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 2D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea2.getImageArea().scaleImage(2);
				hiddenArea3.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 3D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea3.getImageArea().scaleImage(2);

				
				
				hidden2Area.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.MANY_TO_ONE, null, samplingH2, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden2);
				hidden2Area.getImageArea().scaleImage(2);
				
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.ONE_TO_ONE, null, true).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				outArea.getImageArea().scaleImage(2);
				
				net.finalizeConnections();
				
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Color color = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int y = 0; y < height; y+=sampling){
					for(int x = 0; x < width; x+=sampling){
							color = pixelReader.getColor(x, y);
							pixNode = (PixelNode) inArea.getNodeXY(x, y, sampling);
							pixNode.setEntry(color.getOpacity());
					}
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				hiddenArea.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea2.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea3.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);

				
				//net.getNode(1, 0, 5050).getArea().getFilter().filterToString(0);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	@Test
	public void testSimpleMaxPooling(){
		
		showSimpleMaxPooling();
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void showSimpleMaxPooling(){
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				Image image = new Image("file:./resources/icon_48x48.png");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				System.out.println("image :" + width + "x" + height);
				
				int sampling = 1;
				int nbPixels = (int) ((width / sampling) * (height / sampling)) ;
				System.out.println("nb pixels inputs :" + nbPixels);
				
				int samplingH = 2;
				int nbPixelsHidden = (int) ((width / samplingH) * (height / samplingH)) ;
				System.out.println("nb pixels hidden :" + nbPixelsHidden);
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare((width / samplingH), (height / samplingH), true);
				AreaSquare hiddenArea2 = new AreaSquare((width / samplingH), (height / samplingH), true);

				ILayer hiddenLayer = new Layer(hiddenArea, hiddenArea2);
				
				AreaSquare outArea = new AreaSquare(10, 1, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				hiddenArea.configureLinkage(ELinkage.MAX_POOLING, ELinkageBetweenAreas.ONE_TO_MANY, null, 1, false,  3D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea.getImageArea().scaleImage(2);
				hiddenArea2.configureLinkage(ELinkage.MAX_POOLING, ELinkageBetweenAreas.ONE_TO_MANY, null, 1, false,  2D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea2.getImageArea().scaleImage(2);

				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.ONE_TO_ONE, null, true).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				outArea.getImageArea().scaleImage(2);
				
				net.finalizeConnections();
				
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Color color = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int y = 0; y < height; y+=sampling){
					for(int x = 0; x < width; x+=sampling){
							color = pixelReader.getColor(x, y);
							pixNode = (PixelNode) inArea.getNodeXY(x, y, sampling);
							pixNode.setEntry(color.getOpacity());
					}
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
//				hiddenArea.getFilter(FilterLinkage.ID_FILTER_MAX_POOLING).filterToImage(8);
//				hiddenArea2.getFilter(FilterLinkage.ID_FILTER_MAX_POOLING).filterToImage(8);

				
				//net.getNode(1, 0, 5050).getArea().getFilter().filterToString(0);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	@Test
	public void testLogGaborVision(){
		
		showLogGaborVision();
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void showLogGaborVision(){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				Image image = new Image("file:./resources/sinsum13.gif");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				
				int sampling = 1;
				int nbPixels = (int) ((width * height) / Math.pow(sampling, 2));
				System.out.println("nb pixels inputs :" + nbPixels);
				
				int samplingH = 4;
				int nbPixelsHidden = (int) (nbPixels / Math.pow(samplingH, 2));
				System.out.println("nb pixels hidden :" + nbPixelsHidden);
				
				int samplingH2 = 1;
				int nbPixelsHidden2 = 9;//(int) (nbPixelsHidden / Math.pow(samplingH2, 2));
				System.out.println("nb pixels hidden2 :" + nbPixelsHidden2);
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea2 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea3 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea4 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea5 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea6 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea7 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea8 = new AreaSquare(nbPixelsHidden, true);
				
				AreaSquare hiddenArea9 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea10 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea11 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea12 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea13 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea14 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea15 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea16 = new AreaSquare(nbPixelsHidden, true);
				
				AreaSquare hiddenArea17 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea18 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea19 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea20 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea21 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea22 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea23 = new AreaSquare(nbPixelsHidden, true);
				AreaSquare hiddenArea24 = new AreaSquare(nbPixelsHidden, true);
				ILayer hiddenLayer = new Layer(hiddenArea, hiddenArea2, hiddenArea3, hiddenArea4, hiddenArea5, hiddenArea6, hiddenArea7, hiddenArea8, 
						hiddenArea9,
						hiddenArea10,
						hiddenArea11,
						hiddenArea12,
						hiddenArea13,
						hiddenArea14,
						hiddenArea15,
						hiddenArea16,
						hiddenArea17, hiddenArea18, hiddenArea19, hiddenArea20, hiddenArea21, hiddenArea22, hiddenArea23, hiddenArea24);
				
				AreaSquare hidden2Area = new AreaSquare(nbPixelsHidden2, true);
				ILayer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(1);
				ILayer outLayer = new Layer(outArea);
				net.addLayer(inLayer, hiddenLayer, hidden2Layer, outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				hiddenArea.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 1D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea.getImageArea().scaleImage(2);
				hiddenArea2.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 2D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea2.getImageArea().scaleImage(2);
				hiddenArea3.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 3D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea3.getImageArea().scaleImage(2);
				hiddenArea4.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 4D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea4.getImageArea().scaleImage(2);
				hiddenArea5.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH,  false, 8D, 3D, 5D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea5.getImageArea().scaleImage(2);
				hiddenArea6.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 6D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea6.getImageArea().scaleImage(2);
				hiddenArea7.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 7D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea7.getImageArea().scaleImage(2);
				hiddenArea8.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 8D, 1D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea8.getImageArea().scaleImage(2);
				
				
				hiddenArea9.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false,  8D, 3D, 1D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea9.getImageArea().scaleImage(2);
				hiddenArea10.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 2D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea10.getImageArea().scaleImage(2);
				hiddenArea11.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 3D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea11.getImageArea().scaleImage(2);
				hiddenArea12.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 4D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea12.getImageArea().scaleImage(2);
				hiddenArea13.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 5D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea13.getImageArea().scaleImage(2);
				hiddenArea14.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 6D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea14.getImageArea().scaleImage(2);
				hiddenArea15.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 7D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea15.getImageArea().scaleImage(2);
				hiddenArea16.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 8D, 2D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea16.getImageArea().scaleImage(2);
				
				
				hiddenArea17.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 1D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea17.getImageArea().scaleImage(2);
				hiddenArea18.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 2D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea18.getImageArea().scaleImage(2);
				hiddenArea19.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 3D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea19.getImageArea().scaleImage(2);
				hiddenArea20.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 4D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea20.getImageArea().scaleImage(2);
				hiddenArea21.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 5D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea21.getImageArea().scaleImage(2);
				hiddenArea22.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 6D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea22.getImageArea().scaleImage(2);
				hiddenArea23.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 7D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea23.getImageArea().scaleImage(2);
				hiddenArea24.configureLinkage(ELinkage.LOG_GABOR2, ELinkageBetweenAreas.ONE_TO_MANY, null, samplingH, false, 8D, 3D, 8D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden);
				hiddenArea24.getImageArea().scaleImage(2);
				
				
				hidden2Area.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.MANY_TO_ONE, null, samplingH2, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixelsHidden2);
				hidden2Area.getImageArea().scaleImage(2);
				
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.ONE_TO_ONE, null, true).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				
				
				net.finalizeConnections();
				
				
				//Image noir et blanc en input
				// it's a .gif => Brightness (no opacity)
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Color color = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int y = 0; y < height; y+=sampling){
					for(int x = 0; x < width; x+=sampling){
							color = pixelReader.getColor(x, y);
							pixNode = (PixelNode) inArea.getNodeXY(x / sampling, y / sampling);
							pixNode.setEntry(color.getBrightness());
					}
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				hiddenArea.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea2.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea3.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea4.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea5.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea6.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea7.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea8.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				
				hiddenArea9.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea10.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea11.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea12.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea13.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea14.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea15.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea16.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				
				hiddenArea17.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea18.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea19.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea20.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea21.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea22.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea23.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				hiddenArea24.getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
				
				//net.getNode(1, 0, 5050).getArea().getFilter().filterToString(0);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	
	@Test
	public void testLogPolarRetina(){
		
		showLogPolarRetina();
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void showLogPolarRetina(){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				Image image = new Image("file:./resources/lena128.jpg");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				
				int sampling = 1;
				int nbPixels = (int) ((width * height) / Math.pow(sampling, 2));
				System.out.println("nb pixels inputs :" + nbPixels);
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea2 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea, hiddenArea2);
				
				AreaSquare hidden2Area = new AreaSquare(100, true);
				ILayer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(9, true);
				ILayer outLayer = new Layer(outArea);
				net.addLayer(inLayer, hiddenLayer, hidden2Layer, outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				hiddenArea.configureLinkage(ELinkage.LOG_POLAR, ELinkageBetweenAreas.ONE_TO_MANY, null, false, 8D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea.getImageArea().scaleImage(2);
				
				hiddenArea2.configureLinkage(ELinkage.CARTESIAN_TO_POLAR, ELinkageBetweenAreas.ONE_TO_MANY, null, false, 8D, 3D).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea2.getImageArea().scaleImage(2);
				
				
				hidden2Area.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.MANY_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(100);
				hidden2Area.getImageArea().scaleImage(2);
				
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.ONE_TO_ONE, null, true).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(9);
				
				
				net.finalizeConnections();
				
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Color color = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int y = 0; y < height; y+=sampling){
					for(int x = 0; x < width; x+=sampling){
							color = pixelReader.getColor(x, y);
							pixNode = (PixelNode) inArea.getNodeXY(x / sampling, y / sampling);
							pixNode.setEntry(color.getBrightness());
					}
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				//System.out.println(net.getNode(2, 0, 48).getString());
				//net.getNode(1, 0, 5050).getArea().getFilter().filterToString(0);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	
	private void showContoursHorizontaux( final double ox, final double oy, final double mu, final double alpha){


	}
	
	
	@Test
	public void testArchitecture() {
		fail("Not yet implemented");
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
//		assertEquals(1.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		assertEquals(2.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		assertEquals(3.0D, node.getComputedOutput(), 0.0D);
//		node.computeOutput(false);
//		assertEquals(4.0D, node.getComputedOutput(), 0.0D);		
	}

	@Test
	public void testComputeOutput() throws Exception {
		
//		node.computeOutput(false);
//		assertEquals(0.0D, node.getComputedOutput(), 0.0D);
//		
//		inputLink.setUnlinkedValue(2.0D);
//		node.computeOutput(false);
//		assertEquals(2.0D, node.getComputedOutput(), 0.0D);
//		
//		node.setActivationFx(EActivation.SYGMOID_0_1);
//		inputLink.setUnlinkedValue(0.0D);
//		node.computeOutput(false);
//		assertEquals(0.5D, node.getComputedOutput(), 0.0D);
//		
//		node.setActivationFx(EActivation.SYGMOID_1_1);
//		inputLink.setUnlinkedValue(0.0D);
//		node.computeOutput(false);
//		assertEquals(0.0D, node.getComputedOutput(), 0.0D);
	}

	@Test
	public void testPerformActivationFunction() {
		fail("Not yet implemented");
	}

	@Test
	public void testPerformDerivativeFunction() {
		fail("Not yet implemented");
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
