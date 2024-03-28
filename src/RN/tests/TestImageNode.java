package RN.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.Area;
import RN.AreaSquare;
import RN.ENetworkImplementation;
import RN.IArea;
import RN.ILayer;
import RN.Layer;
import RN.Network;
import RN.algoactivations.EActivation;
import RN.linkage.ELinkage;
import RN.linkage.ELinkageBetweenAreas;
import RN.linkage.LaplacianOfGaussianLinkage;
import RN.linkage.OneToOneOctaveAreaLinkage;
import RN.linkage.vision.Histogram;
import RN.linkage.vision.KeyPoint;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.ImageNode;
import RN.nodes.Node;
import RN.nodes.PixelNode;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class TestImageNode extends Application{

	private ImageNode imageNode = null;
	private ImageNode imageNode2 = null;
	
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
	}

	@Test
	public void testDisconnect() {
		assertTrue(imageNode.getInputs().size() > 0);
		assertTrue(imageNode.getOutputs().size() > 0);
		imageNode.disconnect();
		assertEquals(0, imageNode.getInputs().size());
		assertEquals(0, imageNode.getOutputs().size());
	}
	
	@Test
	public void testDrawImageData() {
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				int pixels = 10000;
				int width = (int) Math.sqrt(pixels);
				
				double filterIntensity = 255D;
				double imagePixelsIntensity = 255D;
				
				for (int pixIdx = 0; pixIdx < pixels * 3; pixIdx+=3) {
					
					imageNode.setImageData(pixIdx, (byte)  (255D * Math.exp((double)(-pixIdx - 15000D) / (2D * width * width * 3D)))) ;
					//imageData[idx + 1] =  (byte) ((byte)  filterIntensity & (byte)  imagePixelsIntensity);
					imageNode.setImageData(pixIdx + 1, (byte)  (((float) pixIdx / (pixels * 3)) * filterIntensity) ) ;
					imageNode.setImageData(pixIdx + 2, (byte)  ((1f - (float) pixIdx / (pixels * 3)) * imagePixelsIntensity)) ;
				}

				imageNode.scaleImage(2);
				//imageNode.initImageScene();
				imageNode.drawImageData(null);
				
			}
		});
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	}
	
	@Test
	public void testRotateImageData() {
		
		Image image = new Image("file:./resources/bikes.jpg");
		int pixels = (int) (image.getWidth() * image.getHeight());
		int width = (int) Math.sqrt(pixels);
		PixelReader pixelReader = image.getPixelReader();
		
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				imageNode2 = new ImageNode(EActivation.IDENTITY);
				imageNode2.initImageData(width, width);
				imageNode2.scaleImage(3D);
				imageNode.initImageData(width, width);
				
				boolean isMiddle;
				int[] coord;
				Color color = null;
				
				System.out.println("Begin lecture image ...");


				for(int idx = 0; idx < pixels * 3; idx+=3){
					coord = nodeIdToNodeXY(idx / 3, width);
					color = pixelReader.getColor(coord[0], coord[1]);
					imageNode.setImageData(idx, (byte)  (255D * color.getRed()) ) ;
					imageNode.setImageData(idx + 1, (byte) (255D * color.getGreen())  ) ;
					imageNode.setImageData(idx + 2, (byte)  (255D * color.getBlue()) ) ;
					
				}
				System.out.println("End lecture image.");
				
				for (int pixIdx = 0; pixIdx < pixels * 3; pixIdx+=3) {
					isMiddle = (pixIdx/3) % (width/2) == 0 && (pixIdx/3) % 100 != 0;
					if(isMiddle){
						imageNode.setImageData(pixIdx, (byte)  0 ) ;
						imageNode.setImageData(pixIdx + 1, (byte) 255D ) ;
						imageNode.setImageData(pixIdx + 2, (byte)  0 ) ;
					}
				}
				
				
				for (int pixIdx = 0; pixIdx < pixels * 3; pixIdx+=3) {
						coord = nodeIdToNodeXY(pixIdx / 3, width);
						if(coord[0] >= 45 && coord[0] <= 55 && coord[1] >= 0 && coord[1] <= 20){
							imageNode.setImageData(pixIdx, (byte)  0 ) ;
							imageNode.setImageData(pixIdx + 1, (byte) 255D) ;
							imageNode.setImageData(pixIdx + 2, (byte)  0 ) ;
						}
				}
				
				
//				imageNode.initImageScene();
//				imageNode.drawImageData(null);
				
				
				
			}
		});
		
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		int[] coord;
		int nodeId;
		int previousRow;
		Color color = null;
		
		
		//imageNode2.initImageData();
		byte rByte;
		byte gByte;
		byte bByte;
		

		for (double theta = 0; theta < 2 * Math.PI; theta += (Math.PI / 360D)) {

					for (int pixIdx = 0; pixIdx < pixels * 3; pixIdx += 3) {
						
						coord = nodeIdToNodeXY(pixIdx / 3, width);
						nodeId = nodeXYToNodeId(coord[0], coord[1], width / 2, width / 2, theta, width) * 3;
						//BaseTransform aff = Affine2D.getRotateInstance(theta, coord[0], coord[1]);
						
						rByte = imageNode.getImageData(nodeId);
						gByte = imageNode.getImageData(nodeId + 1);
						bByte = imageNode.getImageData(nodeId + 2);
						
						imageNode2.setImageData(pixIdx, rByte);
						imageNode2.setImageData(pixIdx + 1, gByte);
						imageNode2.setImageData(pixIdx + 2, bByte);
						
						// Je le garde car : A l'orgine un Lissage bilinéaire mais produit un effet spectaculaire du fait de l'erreur imageNode au lieu de imageNode2
//						color = pixelReader.getColor(coord[0], coord[1]);
//						if(pixIdx >= 3){
//							imageNode.setImageData(pixIdx - 3, (byte)  ( (rByte + imageNode.getImageData(pixIdx - 3)) / 2 ) );
//							imageNode.setImageData(pixIdx - 2, (byte) ( (gByte + imageNode.getImageData(pixIdx - 2)) / 2)  );
//							imageNode.setImageData(pixIdx - 1, (byte)  ( (bByte + imageNode.getImageData(pixIdx - 1)) / 2)  );
//						}
//						
//						if(pixIdx >= width*3){
//							previousRow = pixIdx - (width * 3);
//							imageNode.setImageData(previousRow, (byte)  ( (rByte + imageNode.getImageData(previousRow)) / 2 ) );
//							imageNode.setImageData(previousRow + 1, (byte) ( (gByte + imageNode.getImageData(previousRow + 1)) / 2)  );
//							imageNode.setImageData(previousRow + 2, (byte)  ( (bByte + imageNode.getImageData(previousRow + 2)) / 2)  );
//						}

					}

					
					//final Double theto = theta;
					
					Platform.runLater(new Runnable(){

						@Override
						public void run() {
							//imageNode2.getStage().setTitle("Tetha : " + (int) Math.toDegrees(theto));
							imageNode2.drawImageData(null);
						}
					});
					
					
//					try {
//						Thread.sleep(50);
//					} catch (InterruptedException e) {
//						e.printStackTrace();
//					}

		
		}
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	}
	
	
	public int[] nodeIdToNodeXY(int id, int widthPx) {

		int x;
		int y;

		double val = (double) id / widthPx;
		x = (int) (id - (Math.floor(val) * widthPx));
		y = (int) Math.floor(val);

		return new int[] { x, y };

	}
	
	public int nodeXYToNodeId(int x, int y, int x0, int y0, double theta, int widthpx){
		
		int newX = 0;
		int newY = 0;
		
		newX = (int) Math.round((x-x0) * Math.cos(theta) - (y-y0) * Math.sin(theta)) + x0 ;
		newY = (int) Math.round((x-x0) * Math.sin(theta) + (y-y0) * Math.cos(theta)) + y0 ;
		
		newX = Math.min( Math.max(newX, 0), widthpx - 1 );
		newY = Math.min( Math.max(newY, 0), widthpx - 1 );
		
		return newX + newY * widthpx ;
	}
	
	@Test
	public void testShowCenterPointImage() {
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				int nbPixels = 10000;
				
				AreaSquare area = new AreaSquare(nbPixels, true);
				
				PixelNode centerPix = new PixelNode();
				Link in = Link.getInstance(ELinkType.REGULAR, false);
				in.setUnlinkedValue(1D);
				centerPix.addInput(in);
				
				area.addNode(centerPix);
				
				centerPix.setNodeId(5050);
				centerPix.initParameters();
				
				

				assertEquals(50D, centerPix.getX(), 0.0D);
				assertEquals(50D, centerPix.getY(), 0.0D);
				assertEquals(100.0, area.getWidthPx(), 0.0D);
				assertEquals(area.getWidthPx(), area.getHeightPx(), 0.0D);
				
				Area area1 = new Area(1);
				Node hiddenPix = new Node();
				area1.addNode(hiddenPix);
				centerPix.link(hiddenPix, ELinkType.REGULAR);
				
				try {
					centerPix.computeOutput(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				imageNode.scaleImage(2);
				imageNode.showImage(hiddenPix);
				
			}
		});
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
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
		
				Area areaOut = new Area(1);
				Node outNode = new Node();
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

				imageNode.scaleImage(2);
				imageNode.showImage(outNode);
		
			}});
		

	}
	
	/**
	 * Test de la dérivée seconde de gaussienne
	 * ou les différences de luminance dans l'image.
	 * Vision : réalise un champ récépteur ON/OFF, liaisons des photorécépteurs (cones et batonnets) vers un neurone bipolaire puis
	 * vers un neurone ganglionnaire ON/OFF.
	 */
	@Test
	public void testShowGaussianImage() {
		
		
		//int nbPixels = 4;
		double N = 9D;
		double deltaX = 1;
		double k = 13D;
		double Ox = (N-1) * deltaX / k;
		double Oy = Ox;
		double alpha = 1D;
		double Mu = 0D;
		
		showImage( alpha, Ox, Oy, Mu);
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}
	
	private void showImage( final double alpha, final double ox, final double oy, final double mu){
		
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				int nbPixels = 50 * 100;
				
				Network net = Network.getInstance();
				
				AreaSquare areaIn = new AreaSquare(50, 100, true);
				areaIn.getImageArea().scaleImage(2);
				Layer layerIn = new Layer(areaIn);
				
				AreaSquare area = new AreaSquare(50, 100, true);
				area.getImageArea().scaleImage(2);
				ILayer hiddenLayer = new Layer(area);
				
				net.addLayer(layerIn, hiddenLayer);
				
				PixelNode gaussNode = new PixelNode();
				gaussNode.setArea(area);
				gaussNode.setNodeId(5050);
				area.addNode(gaussNode);
				
				LaplacianOfGaussianLinkage linkage = new LaplacianOfGaussianLinkage();
				linkage.setLinkageAreas(ELinkageBetweenAreas.ONE_TO_ONE);

				linkage.setSigmaX(ox);
				linkage.setSigmaY(oy);
				linkage.setAlpha(alpha);
				linkage.setMu(mu);
				area.setLinkage(linkage);
				linkage.setArea(area);
				
				
				PixelNode pixNode = null;
				Link link = null;
				double margeCarre = 15D;
				
				for(int idx = 1; idx <= nbPixels; idx++){
					
					pixNode = new PixelNode();
					areaIn.addNode(pixNode);
					
					link = Link.getInstance(ELinkType.REGULAR, false);
					pixNode.addInput(link);
					
					// les photorécepteurs sont continuellements activés même en l'absence de stimulis -> 0.2D
					if(pixNode.getX() > margeCarre && pixNode.getX() < pixNode.getAreaSquare().getWidthPx() - margeCarre && pixNode.getY() > margeCarre && pixNode.getY() < pixNode.getAreaSquare().getHeightPx() - margeCarre)
						pixNode.setEntry(1D);
					else
						pixNode.setEntry(0D);
					
					
				}
				
				net.finalizeConnections();
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				areaIn.showGradients(15D, 0D, 5, null);
				
				System.out.println(gaussNode.getString());
				
				int idx = 0;
				double somme = 0D;
				for(Link liaison : gaussNode.getInputs()){
					System.out.println(idx++ + " w"+ liaison.getSourceNode().getNodeId() + " : "+ liaison.getWeight());
					somme += liaison.getWeight();
				}
				System.out.println("Somme :" + somme);

				//areaIn.showImageArea();
				//area.showImageArea();
				
		
			}});
		

	}
	

	
	@Test
	public void testShowConvolutionImage() {
		
		
		showConvolutionImage();
		
		try {
			Thread.sleep(15000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}
	

	
	private void showConvolutionImage(){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				
				Image image = new Image("file:./resources/circle-square-1.png");
				int width = (int) image.getWidth();
				int height = (int) image.getHeight();
				int nbPixels = width * height;
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(width, height, true);
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels/4, true);
				hiddenArea.configureLinkage(ELinkage.LOG_STATIC, null, 2, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels/4);
				hiddenArea.getImageArea().scaleImage(2);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
				AreaSquare outArea = new AreaSquare(1);
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode(false, ENodeType.PIXEL).createNodes(1);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, outLayer);
				
				
				net.finalizeConnections();
				
				
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
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
				
				
				inArea.showGradients(10D, 0D, 1, null);
				
				//System.out.println(net.getString());
				
		
			}});

	}
	
	@Test
	public void testShowConvolutionGaussianImage() {
		
		
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
		
		
		//showLearningConvolutionGaussianImage(Ox, Oy, Mu, alpha);
		//showConvolutionGaussianImage(Ox, Oy, Mu, alpha);
		//showMultiScale(Ox, Oy, Mu, alpha);
		showMultiScale0(Ox, Oy, Mu, alpha);
		//showMultiScale2(Ox, Oy, Mu, alpha);
		//showDetectionCoinsImage(Ox, Oy, Mu, alpha);
			

		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}
	
	private void showLearningConvolutionGaussianImage( final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				Image image = new Image("file:./resources/icon_128x128.png");
				int nbPixels = (int) (image.getWidth() * image.getHeight());
				
				
				Network net = Network.getInstance(ENetworkImplementation.LINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
				AreaSquare hidden2Area = new AreaSquare(nbPixels / 4, true);
				ILayer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(625, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, hidden2Layer, outLayer);
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				
				double k = 1.106D;
				double f = 6D;
				hiddenArea.configureLinkage(ELinkage.SONAG, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea.getImageArea().scaleImage(2);
				
				hidden2Area.configureLinkage(ELinkage.MAX_POOLING, null, false, 3D, 2D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels / 4);
				hidden2Area.getImageArea().scaleImage(2);
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(625);
				outArea.getImageArea().scaleImage(2);
				
				net.finalizeConnections();
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				PixelReader pixelReader = image.getPixelReader();
				Color color = null;
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getOpacity());
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				//System.out.println(net.getNode(1, 0, 5050).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				//System.out.println(net.getString());
		
			}});

	}
	
	private void showConvolutionGaussianImage( final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				Image image = new Image("file:./resources/icon_128x128.png");
				int nbPixels = (int) (image.getWidth() * image.getHeight());
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				AreaSquare in2Area = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea, in2Area);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				AreaSquare hidden2Area = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
				
				
				AreaSquare outArea = new AreaSquare(nbPixels, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, outLayer);
				
//				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(nbPixels);
//				inArea.getImageArea().scaleImage(2);
//				inArea.initBiasWeights(-0.5D);
//				
//				in2Area.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(true, EActivation.SYGMOID_0_1_INVERSE, ENodeType.PIXEL).createNodes(nbPixels);
//				in2Area.getImageArea().scaleImage(2);
//				inArea.initBiasWeights(0.5D);
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				inArea.getImageArea().scaleImage(2);
				in2Area.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.NEGATIVE, ENodeType.PIXEL).createNodes(nbPixels);
				in2Area.getImageArea().scaleImage(2);
				
				
				double k = 1.106D;
				double f = 2D;
				hiddenArea.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.MANY_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea.getImageArea().scaleImage(2);
				
//				hidden2Area.configureLinkage(ELinkage.ONE_TO_ONE_FETCH_AREA, null, false).configureNode(false, EActivation.NEGATIVE, ENodeType.GAUSSIAN).createNodes(nbPixels);
//				hidden2Area.getImageArea().scaleImage(2);
				
				outArea.configureLinkage(ELinkage.ONE_TO_ONE_FETCH_OCTAVE_AREA, null, false).configureNode( false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				outArea.getImageArea().scaleImage(2);
				
				
				//net.finalizeConnections();
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				PixelNode pix2Node = null;
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					pix2Node = (PixelNode) in2Area.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getOpacity());
					pix2Node.setEntry(color.getOpacity());
				}
				System.out.println("End lecture image.");
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				//System.out.println(net.getNode(1, 0, 5050).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

	}
	
	private void showMultiScale(final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				Image image = new Image("file:./resources/square-2.png");
				int nbPixels = (int) (image.getWidth() * image.getHeight());
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea_1_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea_1_0, hiddenArea_1_1, hiddenArea_1_2, hiddenArea_1_3, hiddenArea_1_4);
				
				
				AreaSquare hiddenArea_2_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_4 = new AreaSquare(nbPixels, true);
				hiddenLayer.addAreas(hiddenArea_2_0, hiddenArea_2_1, hiddenArea_2_2, hiddenArea_2_3, hiddenArea_2_4);
				
				
				AreaSquare hiddenArea_3_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_3_4 = new AreaSquare(nbPixels, true);
				hiddenLayer.addAreas(hiddenArea_3_0, hiddenArea_3_1, hiddenArea_3_2, hiddenArea_3_3, hiddenArea_3_4);

				Area outArea = new AreaSquare(nbPixels, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, outLayer);
				
				
				PixelNode pixNode = null;
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false);
				inArea.configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				double t = 0D;
				double f = Math.pow(2D, t);
				double k = 1.106D;
				//double k = 1.6D;
				hiddenArea_1_0.configureLinkage(ELinkage.DOG, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_1.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_2.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_3.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_4.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				t = 1D;
				f = Math.pow(2D, t);
				hiddenArea_2_0.configureLinkage(ELinkage.DOG, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_1.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_2.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_3.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_4.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				t = 2D;
				f = Math.pow(2D, t);
				hiddenArea_3_0.configureLinkage(ELinkage.DOG, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_3_1.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_3_2.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_3_3.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_3_4.configureLinkage(ELinkage.DOG, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
			
				outArea.configureLinkage(ELinkage.ONE_TO_ONE_FETCH_OCTAVE_AREA, null, false).configureNode( false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				long start = System.currentTimeMillis();
				net.finalizeConnections();
				long stop = System.currentTimeMillis();
				System.out.println("finalize Connections : " + (stop - start) / 1000l + " secondes");
				
				//Image noir et blanc en input
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.grayscale().getBrightness());
					
				}
				
				System.out.println("Begin propagation...");
				start = System.currentTimeMillis();
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				stop = System.currentTimeMillis();
				System.out.println("End propagation.");
				System.out.println("Propagation : " + (stop - start) / 1000l + " secondes");
				
				inArea.imageToString();
				hiddenArea_1_0.imageToString();
				hiddenArea_1_1.imageToString();
				hiddenArea_1_2.imageToString();
				hiddenArea_1_3.imageToString();
				hiddenArea_1_4.imageToString();
				
				//System.out.println(net.getNode(3, 0, 0).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(2, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_V1Orientation);
		
			}});

	}
	
	private void showMultiScale0(final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
				
				Image image = new Image("file:./resources/square-2.png");
				int nbPixels = (int) (image.getWidth() * image.getHeight());
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea_0_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer0 = new Layer(hiddenArea_0_0, hiddenArea_0_1, hiddenArea_0_2, hiddenArea_0_3, hiddenArea_0_4);
				
				//------------- 
				
				AreaSquare hiddenArea_1_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer1 = new Layer(hiddenArea_1_0, hiddenArea_1_1, hiddenArea_1_2, hiddenArea_1_3, hiddenArea_1_4);
				
				
				//------------- 
				
				AreaSquare hiddenArea_2_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer2 = new Layer(hiddenArea_2_0, hiddenArea_2_1, hiddenArea_2_2, hiddenArea_2_3, hiddenArea_2_4);
				
				AreaSquare outArea = new AreaSquare(nbPixels, true);
				ILayer outLayer = new Layer(outArea);
				
				
				net.addLayer(inLayer, hiddenLayer0, hiddenLayer1, hiddenLayer2, outLayer);
				
				PixelNode pixNode = null;
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				double t = 0D;
				double f = Math.pow(2D, t);
				//double k = 1.106D;
				double k = 1.3D;
				//double k = 1.6D;
				hiddenArea_0_0.configureLinkage(ELinkage.GAUSSIAN, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_1.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_2.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_3.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_4.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				hiddenArea_1_0.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_1.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_2.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_3.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_4.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				hiddenArea_2_0.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_1.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_2.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_3.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_4.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				
				
				long start = System.currentTimeMillis();
				net.finalizeConnections();
				long stop = System.currentTimeMillis();
				System.out.println("finalize Connections : " + (stop - start) / 1000l + " secondes");
				
				//Image noir et blanc en input
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getBrightness());
				}
				
				System.out.println("Begin propagation...");
				start = System.currentTimeMillis();
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				stop = System.currentTimeMillis();
				System.out.println("End propagation.");
				System.out.println("Propagation : " + (stop - start) / 1000l + " secondes");
				
				
				List<KeyPoint> kpList1 = new ArrayList<KeyPoint>();
				for(IArea area : hiddenLayer2.getAreas()){
					kpList1.addAll(((OneToOneOctaveAreaLinkage) area.getLinkage()).getKeyPoints());
				}
				System.out.println("\n SQUARE 1 : nb key points: "+ kpList1.size());
				
				image = new Image("file:./resources/square-3.png");
				
				pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getBrightness());
				}
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				List<KeyPoint> kpList2 = new ArrayList<KeyPoint>();
				for(IArea area : hiddenLayer2.getAreas()){
					kpList2.addAll(((OneToOneOctaveAreaLinkage) area.getLinkage()).getKeyPoints());
				}

				System.out.println("\n SQUARE 3 : nb key points: "+ kpList2.size());
				
//				inArea.imageToString();
//				hiddenArea_0_0.imageToString();
//				inArea.compareArea(hiddenArea_0_0);
				
//				hiddenArea_1_0.imageToString();
//				hiddenArea_1_1.imageToString();
//				hiddenArea_1_2.imageToString();
//				hiddenArea_1_3.imageToString();
//				hiddenArea_1_4.imageToString();
				
				//hiddenArea_0_0.compareArea(hiddenArea_0_6);
				
			}
			
		});
	}
	
	private void showMultiScale2(final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				Image image = new Image("file:./resources/e-16x16.png");
				int nbPixels = (int) (image.getWidth() * image.getHeight());
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea_0_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer0 = new Layer();
				hiddenLayer0.addAreas(hiddenArea_0_0, hiddenArea_0_1, hiddenArea_0_2, hiddenArea_0_3, hiddenArea_0_4);
				
				
				AreaSquare hiddenArea_0_5 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_6 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_7 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_8 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_9 = new AreaSquare(nbPixels, true);
				hiddenLayer0.addAreas(hiddenArea_0_5, hiddenArea_0_6, hiddenArea_0_7, hiddenArea_0_8, hiddenArea_0_9);
				
				
				AreaSquare hiddenArea_0_10 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_11 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_12 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_13 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_0_14 = new AreaSquare(nbPixels, true);
				hiddenLayer0.addAreas(hiddenArea_0_10, hiddenArea_0_11, hiddenArea_0_12, hiddenArea_0_13, hiddenArea_0_14);
				
				//------------- 
				
				AreaSquare hiddenArea_1_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_4 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer1 = new Layer(hiddenArea_1_0, hiddenArea_1_1, hiddenArea_1_2, hiddenArea_1_3, hiddenArea_1_4);
				
				
				AreaSquare hiddenArea_1_5 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_6 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_7 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_8 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_9 = new AreaSquare(nbPixels, true);
				hiddenLayer1.addAreas(hiddenArea_1_5, hiddenArea_1_6, hiddenArea_1_7, hiddenArea_1_8, hiddenArea_1_9);
				
				
				AreaSquare hiddenArea_1_10 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_11 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_12 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_13 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_1_14 = new AreaSquare(nbPixels, true);
				hiddenLayer1.addAreas(hiddenArea_1_10, hiddenArea_1_11, hiddenArea_1_12, hiddenArea_1_13, hiddenArea_1_14);
				
				//------------- 
				
				AreaSquare hiddenArea_2_0 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_1 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_2 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_3 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_4 = new AreaSquare(nbPixels, true);
				
				AreaSquare hiddenArea_2_5 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_6 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_7 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_8 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_9 = new AreaSquare(nbPixels, true);
				
				AreaSquare hiddenArea_2_10 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_11 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_12 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_13 = new AreaSquare(nbPixels, true);
				AreaSquare hiddenArea_2_14 = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer2 = new Layer(hiddenArea_2_0, hiddenArea_2_1, hiddenArea_2_2, hiddenArea_2_3, hiddenArea_2_4);
				hiddenLayer2.addAreas(hiddenArea_2_5, hiddenArea_2_6, hiddenArea_2_7, hiddenArea_2_8, hiddenArea_2_9);
				hiddenLayer2.addAreas(hiddenArea_2_10, hiddenArea_2_11, hiddenArea_2_12, hiddenArea_2_13, hiddenArea_2_14);
				
				//------------- 
				int pixCountL3 = 625;
				
				AreaSquare hiddenArea_3_0 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_1 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_2 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_3 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_4 = new AreaSquare(pixCountL3, true);
				
				AreaSquare hiddenArea_3_5 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_6 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_7 = new AreaSquare(pixCountL3, true);
				AreaSquare hiddenArea_3_8 = new AreaSquare(pixCountL3, true);

				
				ILayer hiddenLayer3 = new Layer(hiddenArea_3_0, hiddenArea_3_1, hiddenArea_3_2, hiddenArea_3_3, hiddenArea_3_4);
				hiddenLayer3.addAreas(hiddenArea_3_5, hiddenArea_3_6, hiddenArea_3_7, hiddenArea_3_8);

				AreaSquare outArea = new AreaSquare(nbPixels, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer0, hiddenLayer1, hiddenLayer2, hiddenLayer3, outLayer);
				
				
				PixelNode pixNode = null;
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				double t = 0D;
				double f = Math.pow(2D, t);
				//double k = 1.106D;
				double k = 1.3D;
				//double k = 1.6D;
				hiddenArea_0_0.configureLinkage(ELinkage.GAUSSIAN, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_1.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_2.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_3.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_4.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				t = 1D;
				f = Math.pow(2D, t);
				hiddenArea_0_5.configureLinkage(ELinkage.GAUSSIAN, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_6.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_7.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_8.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_9.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				t = 2D;
				f = Math.pow(2D, t);
				hiddenArea_0_10.configureLinkage(ELinkage.GAUSSIAN, null, false, f * k).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_11.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 2D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_12.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 3D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_13.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 4D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_0_14.configureLinkage(ELinkage.GAUSSIAN, null, false, f * Math.pow(k, 5D)).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				
				hiddenArea_1_0.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_1.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE,null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_2.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_3.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_4.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				hiddenArea_1_5.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_6.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_7.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_8.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_9.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				hiddenArea_1_10.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_11.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_12.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_13.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_1_14.configureLinkage(ELinkage.ONE_TO_ONE_FILTER, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				
				hiddenArea_2_0.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_1.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_2.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_3.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_4.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				hiddenArea_2_5.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_6.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_7.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_8.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_9.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				hiddenArea_2_10.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_11.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_12.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_13.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hiddenArea_2_14.configureLinkage(ELinkage.ONE_TO_ONE_OCTAVE, ELinkageBetweenAreas.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				
				
				hiddenArea_3_0.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_1.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_2.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_3.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_4.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				
				hiddenArea_3_5.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_6.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_7.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);
				hiddenArea_3_8.configureLinkage(ELinkage.MANY_TO_MANY_OCTAVE, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(pixCountL3);

				
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, ELinkageBetweenAreas.MANY_TO_ONE, null, true).configureNode(true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(10);
				
				
				long start = System.currentTimeMillis();
				net.finalizeConnections();
				long stop = System.currentTimeMillis();
				System.out.println("finalize Connections : " + (stop - start) / 1000l + " secondes");
				
				//Image noir et blanc en input
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getOpacity());
				}
				
				System.out.println("Begin propagation...");
				start = System.currentTimeMillis();
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				stop = System.currentTimeMillis();
				System.out.println("End propagation.");
				System.out.println("Propagation : " + (stop - start) / 1000l + " secondes");
				
				
				List<KeyPoint> kpList1 = new ArrayList<KeyPoint>();
				for(IArea area : hiddenLayer2.getAreas()){
					kpList1.addAll(((OneToOneOctaveAreaLinkage) area.getLinkage()).getKeyPoints());
				}
				System.out.println("\n SQUARE 1 : nb key points: "+ kpList1.size());
				
				image = new Image("file:./resources/e-translated-16x16.png");
				
				pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getOpacity());
				}
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				List<KeyPoint> kpList2 = new ArrayList<KeyPoint>();
				for(IArea area : hiddenLayer2.getAreas()){
					kpList2.addAll(((OneToOneOctaveAreaLinkage) area.getLinkage()).getKeyPoints());
				}

				System.out.println("\n SQUARE 3 : nb key points: "+ kpList2.size());
				boolean keyPointMatches = true;
				for(KeyPoint kp1 : kpList1){
					for(Histogram histo1 : kp1.getDescriptor().getHistograms()){
						
						for(KeyPoint kp2 : kpList2){
							
							for(Histogram histo2 : kp2.getDescriptor().getHistograms()){
								
								if(histo1.getMaxOrientationKey() != null && histo2.getMaxOrientationKey() != null && histo1.getMaxOrientationKey() != histo2.getMaxOrientationKey()){
									keyPointMatches = false;
								}
								
							}
							if(keyPointMatches){
								System.out.println(kp1.getKeyPointNode() + " matches with keyPoint : " + kp2.getKeyPointNode());
							}
						}
					}
				}
				
				
//				hiddenArea_1_0.imageToString();
//				hiddenArea_1_1.imageToString();
//				hiddenArea_1_2.imageToString();
//				hiddenArea_1_3.imageToString();
//				hiddenArea_1_4.imageToString();
				
				//hiddenArea_0_0.imageToString();
				//hiddenArea_0_14.imageToString();
				
				//hiddenArea_0_0.compareArea(hiddenArea_0_6);
				
				//System.out.println(net.getString());
				
				//System.out.println(net.getNode(3, 0, 0).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG);
				//net.getNode(2, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_V1Orientation);
		
			}});

	}
	
	@Test
	public void testArchitecture() {
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				//Image noir et blanc en input
				Image image = new Image("file:D:\\git\\deepnet\\resources\\bikes.jpg");
				
				int nbPixels = (int) (image.getHeight() * image.getWidth());
				
				
				Network net = Network.getInstance(ENetworkImplementation.UNLINKED);
				
				AreaSquare inArea = new AreaSquare(nbPixels, true);
				Layer inLayer = new Layer(inArea);
				
				AreaSquare hiddenArea = new AreaSquare(nbPixels, true);
				ILayer hiddenLayer = new Layer(hiddenArea);
				
				AreaSquare hidden_1_0_Area = new AreaSquare(nbPixels, true);
				AreaSquare hidden_1_1_Area = new AreaSquare(nbPixels, true);
				AreaSquare hidden_1_2_Area = new AreaSquare(nbPixels, true);
				AreaSquare hidden_1_3_Area = new AreaSquare(nbPixels, true);
				AreaSquare hidden_1_4_Area = new AreaSquare(nbPixels, true);
				ILayer hidden1Layer = new Layer(hidden_1_0_Area, hidden_1_1_Area, hidden_1_2_Area, hidden_1_3_Area);
				
				AreaSquare hidden2Area = new AreaSquare(nbPixels, true);
				ILayer hidden2Layer = new Layer(hidden2Area);
				
				AreaSquare outArea = new AreaSquare(9, true);
				ILayer outLayer = new Layer(outArea);
				
				net.addLayer(inLayer, hiddenLayer, hidden1Layer, hidden2Layer, outLayer);
				
				
				PixelNode pixNode = null;
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL)
						.createNodes(nbPixels);
				
				hiddenArea.configureLinkage(ELinkage.LOG, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL)
							.createNodes(nbPixels);
				
				hidden_1_0_Area.configureLinkage(ELinkage.LOG_GABOR2, null, false, 8D, 3D, 1D, 2D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hidden_1_1_Area.configureLinkage(ELinkage.LOG_GABOR2, null, false, 8D, 3D, 2D, 2D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hidden_1_2_Area.configureLinkage(ELinkage.LOG_GABOR2, null, false, 8D, 3D, 3D, 2D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				hidden_1_3_Area.configureLinkage(ELinkage.LOG_GABOR2, null, false, 8D, 3D, 4D, 2D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
//				hidden_1_1_Area.configureLinkage(ELinkage.V1_ORIENTATIONS, null, false, 45D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
//				hidden_1_2_Area.configureLinkage(ELinkage.V1_ORIENTATIONS, null, false, 90D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
//				hidden_1_3_Area.configureLinkage(ELinkage.V1_ORIENTATIONS, null, false, 135D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
//				hidden_1_4_Area.configureLinkage(ELinkage.V1_ORIENTATIONS, null, false, 180D).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				hidden2Area.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.MANY_TO_ONE, null, false).configureNode(true, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(nbPixels);
				
				outArea.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode( true, EActivation.SYGMOID_0_1, ENodeType.PIXEL).createNodes(1);
				
				
				net.finalizeConnections();
				
				PixelReader pixelReader = image.getPixelReader();
				Color color = null;
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.setEntry(color.getBrightness());
					
				}
				
				
				System.out.println("Begin propagation...");
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				System.out.println("End propagation.");
				
				
//				System.out.println(net.getNode(3, 0, 0).getString());
//				net.getNode(1, 0, 5050).getArea().getFilter(FilterLinkage.ID_FILTER_LOG).filterToString();
//				net.getNode(2, 0, 5050).getArea().getFilter(FilterLinkage.ID_FILTER_V1Orientation).filterToString();
				
			}});
		
		try {
			Thread.sleep(6000000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void showDetectionCoinsImage( final double ox, final double oy, final double mu, final double alpha){
		Platform.runLater(new Runnable(){

			@Override
			public void run() {
		
				int nbPixels = 10000;
				int pixelGaussianNodes = 10000;
				
				ImageNode imageNode0 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode1 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode2 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode3 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode4 = new ImageNode(EActivation.IDENTITY);
				ImageNode imageNode5 = new ImageNode(EActivation.IDENTITY);
				
				Network net = Network.getInstance();
				
				AreaSquare inArea = new AreaSquare(nbPixels, true, "Input");
				Layer inLayer = new Layer(inArea);
				net.addLayer(inLayer);
				
				AreaSquare hiddenArea_1 = new AreaSquare(nbPixels, true, "Simple contour");
				Layer hiddenLayer_1 = new Layer(hiddenArea_1);
				net.addLayer(hiddenLayer_1);
				
				AreaSquare hiddenArea_2_0 = new AreaSquare(nbPixels, true, "Pass by 0 - X");
				AreaSquare hiddenArea_2_1 = new AreaSquare(nbPixels, true, "Pass by 0 - Y");
				Layer hiddenLayer_2 = new Layer(hiddenArea_2_0, hiddenArea_2_1);
				net.addLayer(hiddenLayer_2);
				
				AreaSquare hiddenArea_3 = new AreaSquare(nbPixels, true, "One to one fetch area - regular node");
				Layer hiddenLayer_3 = new Layer(hiddenArea_3);
				net.addLayer(hiddenLayer_3);
				
				Area outArea = new Area(1);
				Layer outLayer = new Layer(outArea);
				net.addLayer(outLayer);
				
				
				inArea.configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false, ENodeType.PIXEL).createNodes(nbPixels);
				
				hiddenArea_1.configureLinkage(ELinkage.LOG, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);
				
				hiddenArea_2_0.configureLinkage(ELinkage.FIRST_DERIVATED_GAUSSIAN, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);
				hiddenArea_2_0.getImageArea().scaleImage(2);
				
				hiddenArea_2_1.configureLinkage(ELinkage.FIRST_DERIVATED_GAUSSIAN, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);
				hiddenArea_2_1.getImageArea().scaleImage(2);
				
				//hiddenArea_2.configureNode(false, EActivation.IDENTITY, ELinkage.GENERIC, false, ENodeType.PASSBY0).configureLinkageFunction(ESamples.G_Dxy_DE_MARR).createNodes(pixelGaussianNodes);
				
				hiddenArea_3.configureLinkage(ELinkage.ONE_TO_ONE, ELinkageBetweenAreas.MANY_TO_ONE, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(pixelGaussianNodes);
				hiddenArea_2_1.getImageArea().scaleImage(2);

				outArea.configureLinkage(ELinkage.MANY_TO_MANY, null, false).configureNode(false, EActivation.IDENTITY, ENodeType.PIXEL).createNodes(1);
				
				net.finalizeConnections();
				
				//Image noir et blanc en input
				System.out.println("Begin lecture image ...");
				PixelNode pixNode = null;
				Image image = new Image("file:./resources/logo-transparent.png");
				PixelReader pixelReader = image.getPixelReader();
				for(int idx = 0; idx < nbPixels; idx++){
					pixNode = (PixelNode) inArea.getNode(idx);
					Color color = pixelReader.getColor(pixNode.getX(), pixNode.getY());
					pixNode.getInput(0).setUnlinkedValue(color.getBrightness());
					
				}
				System.out.println("End lecture image.");
				
//				net.getNode(0, 0, 5050).setInnerNode(imageNode0);
//				net.getNode(1, 0, 5050).setInnerNode(imageNode1);
//				net.getNode(2, 0, 5050).setInnerNode(imageNode2);
//				net.getNode(2, 1, 5050).setInnerNode(imageNode3);
//				net.getNode(2, 2, 0).setInnerNode(imageNode4);
//				net.getNode(3, 0, 5050).setInnerNode(imageNode5);
//				net.getNode(4, 0, 0).setInnerNode(imageNode);
				
				try {
					net.propagation(false);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				//System.out.println(net.getNode(1, 0, 5050).getString());
				//System.out.println(net.getNode(2, 0, 5050).getString());
				System.out.println(net.getNode(3, 0, 5050).getString());
				//net.getNode(1, 0, 5050).getArea().filterToString(FilterLinkage.ID_FILTER_LOG_STATIC);
				//hiddenArea_1.imageToString();
				hiddenArea_3.imageToString();
				//hiddenArea_2_0.compareArea(hiddenArea_2_1);
				
				//net.getNode(1, 0, 5050).getArea().filterToString(1);
				
		
			}});

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
