package RN.linkage;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import RN.IArea;
import RN.IAreaSquare;
import RN.ILayer;
import RN.Identification;
import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.Weight;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.ImageNode;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * @author ericmarchand
 * 
 */
public class ConvolutionLinkage extends FilterLinkage {
	
	// Matrice carrée d'ordre N
	//private static int N = 5;

	Integer stride = null;
	Integer filterWidth = null;
	Integer toCenter = null;

	private static Map<Identification,Weight[][]> sharedWeights = null;
	private static Map<Identification,Weight> sharedBiasWeight = null;

	public ConvolutionLinkage() {
	}

	public ConvolutionLinkage(Integer sampling) {
		this.sampling = sampling;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.linkage.ILinkage#initParameters()
	 */
	public void initParameters() {

		if (params[0] != null)
			filterWidth = params[0].intValue();

		if (params[1] != null)
			stride = params[1].intValue();

		toCenter = ((filterWidth - 1) / 2);
		
		sharedWeights = null;
		sharedBiasWeight = null;

	}
	
	private void initSharedWeights(){
		
		if (sharedWeights == null || sharedWeights.get(getArea().getIdentification()) == null) {
			sharedWeights = new HashMap<Identification, Weight[][]>();
			sharedBiasWeight = new HashMap<Identification, Weight>();
			Weight[][] weights = null;
			Weight biasWeight = null;
			for (IArea area : getArea().getLayer().getAreas()) {
				weights = new Weight[filterWidth][filterWidth];
				biasWeight = new Weight();
				for (int y = 0; y < filterWidth; y++) {
					for (int x = 0; x < filterWidth; x++) {
						weights[x][y] = new Weight();
					}
				}
				sharedWeights.put(area.getIdentification(), weights);
				sharedBiasWeight.put(area.getIdentification(), biasWeight);
				for(INode node : area.getNodes()){
					node.setBiasWeight(biasWeight);
				}
			}
		}
		
	}
	
	public static Weight[][] getSharedFilter(IArea area){
		return sharedWeights.get(area.getIdentification());
	}

	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode) {

		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		IPixelNode pix = (IPixelNode) thisNode;

		IAreaSquare subArea = null;
		IPixelNode centerPix = null;

		List<IPixelNode> nodesInSquare = null;
		Integer startX = null;
		Integer startY = null;
		int paddingX = 0;
		int paddingY = 0;
		
		initSharedWeights();

		for (IArea area : getLinkedAreas()) {

			subArea = (IAreaSquare) area;
			paddingX = (subArea.getWidthPx() - ((IAreaSquare) getArea()).getWidthPx()) / 2;
			paddingY = (subArea.getHeightPx() - ((IAreaSquare) getArea()).getHeightPx()) / 2;
			centerPix = subArea.getNodeXY(pix.getX() * stride + paddingX, pix.getY() * stride + paddingY);
			startX = centerPix.getX() - toCenter;
			startY = centerPix.getY() - toCenter;

			nodesInSquare = subArea.getNodesInSquareZone(startX, startY, filterWidth, filterWidth);
			for (IPixelNode innerPix : nodesInSquare) {
				sigmaWI.sum(innerPix.getComputedOutput() * getLinkAndPutIfAbsent(thisNode, (INode) innerPix, isWeightModifiable(),
						sharedWeights.get(getArea().getIdentification())[innerPix.getX() - startX][innerPix.getY() - startY]).getWeight());
			}

			sigmaWI.sum(-thisNode.getBiasWeightValue());

		}

		return sigmaWI.value();
	}

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {

		IPixelNode pix = (IPixelNode) thisNode;

		IAreaSquare subArea = null;
		IPixelNode centerPix = null;

		List<IPixelNode> nodesInSquare = null;
		Integer startX = null;
		Integer startY = null;
		
		initSharedWeights();

		for (IArea area : getLinkedAreas()) {

			subArea = (IAreaSquare) area;
			centerPix = subArea.getNodeXY(pix.getX() * stride, pix.getY() * stride);
			startX = centerPix.getX() - toCenter;
			startY = centerPix.getY() - toCenter;
			nodesInSquare = subArea.getNodesInSquareZone(startX, startY, filterWidth, filterWidth);
			for (IPixelNode innerPix : nodesInSquare) {
				innerPix.link(thisNode, isWeightModifiable(), sharedWeights.get(getArea().getIdentification())[innerPix.getX() - startX][innerPix.getY() - startY]);
			}
			
		}

	}
	
//	@Override
//	public double getUnLinkedSigmaPotentials(INode thisNode){
//
//		// somme des entrees pondérées
//		SigmaWi sigmaWI = new SigmaWi();
//
//		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
//		
//		initFilter(this, ID_FILTER_CONVOLUTION, ESamples.RAND, (IPixelNode) thisNode, subArea);
//		
//		subArea.applyConvolutionFilter(this, ID_FILTER_CONVOLUTION, (IPixelNode) thisNode, sigmaWI);
//			
//		
//		return sigmaWI.value();
//	}
//
//
//	
//	@Override
//	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
//		
//		
//		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
//		
//		initFilter(this, ID_FILTER_CONVOLUTION, ESamples.RAND, (IPixelNode) thisNode, subArea);
//		
//		subArea.applyConvolutionFilter(this, ID_FILTER_CONVOLUTION, (IPixelNode) thisNode, 0.0000001f);
//
//	}	

	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {

//		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
//		
//		int x = ((IPixelNode) sublayerNode).getX();
//		int y = ((IPixelNode) sublayerNode).getY();
//		
//		int centerX = subArea.getNodeCenterXY().getX();
//		int centerY = subArea.getNodeCenterXY().getY();
//		
//		int Xmat = x - centerX + ( (N-1) / 2 );
//		int Ymat = y - centerY + ( (N-1) / 2 );
//		
//		if(Xmat >= 0 && Xmat <= (N-1) && Ymat >= 0 && Ymat <= (N-1)){
//			return InputSample.getInstance().compute(filterFunction) * 10D - 5D;
//		}
		
		return 0D;
	}
	
	public void kernelToImage(Integer scale){
		
		Weight[][] weights = sharedWeights.get(getArea().getIdentification());
		if(weights != null){
		ImageNode img = new ImageNode(EActivation.IDENTITY, weights[0].length, weights.length);
		
		img.getStage().setTitle("Shared weights #"+ getArea().getIdentification() +" : "+ weights[0].length + " x " + weights.length);
		
		if(scale != null && scale > 1)
			img.scaleImage(scale);
		
		img.insertDataArray(weights);
		img.drawImageData(null);
		}else{
			System.out.println("you have to do at least one feed forward propagation to fill ");
		}
	}
	
	public void addGraphicInterface(Pane pane) {
		
		Button showFilterBtn = new Button("show filter");
		
		showFilterBtn.setTooltip(new Tooltip("show filter"));
		
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(showFilterBtn);
		
		showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

			@Override
			public void handle(ActionEvent event) {
				try {
					kernelToImage(8);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		
		pane.getChildren().addAll(hbox);
		
	}

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT) {
		// TODO Auto-generated method stub

	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer) {
		// TODO Auto-generated method stub

	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT) {
		// TODO Auto-generated method stub

	}

}
