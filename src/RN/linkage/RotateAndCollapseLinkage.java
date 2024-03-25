package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import javafx.scene.layout.Pane;

public class RotateAndCollapseLinkage extends FilterLinkage {
	

	
	protected Double theta = null;
	


	public RotateAndCollapseLinkage() {
	}
	

	
	public void initParameters() {
		
//		if(params.length != 2 && params.length != 3)
//			throw new RuntimeException("Missing RotateAndCollapse parameters'");
//		
//		
//		setTheta(params[0]);		

	}

	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		
		double distance = Math.sqrt(Math.pow(pix.getX(), 2) + Math.pow(pix.getY(), 2));
		
		boolean isShortestPath = false;
		Double minDistance = null;
		double dist = 0D;
		IPixelNode nearestMinMaxNode = null;
		for(IPixelNode maxMinPix : MaxLinkage.getMaxMinNodes()){
			dist = pix.distance(maxMinPix);
			if(minDistance == null || dist < minDistance){
				minDistance = dist;
				nearestMinMaxNode = maxMinPix;
			}
		}
		
		
		
		IPixelNode subPix = subArea.getNodeXY(pix.getX(), pix.getY());
		if(subPix != null){
			sigmaWI.sum(subPix.getComputedOutput());
		}
		
		
		return sigmaWI.value();
	}
	

	




	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		
		double distance = Math.sqrt(Math.pow(pix.getX(), 2) + Math.pow(pix.getY(), 2));
		
		boolean isShortestPath = false;
		Double minDistance = null;
		double dist = 0D;
		IPixelNode nearestMinMaxNode = null;
		for(IPixelNode maxMinPix : MaxLinkage.getMaxMinNodes()){
			dist = pix.distance(maxMinPix);
			if(minDistance == null || dist < minDistance){
				minDistance = dist;
				nearestMinMaxNode = maxMinPix;
			}
		}
		
		IPixelNode subPix = subArea.getNodeXY(pix.getX(), pix.getY());
		if(subPix != null){
			subPix.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
		}
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre Log-Gabor
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
	
				(double) (sublayerNode.getX() - subArea.getWidthPx() / 2D) * 4D,
				(double) (sublayerNode.getY() - subArea.getHeightPx() / 2D) * 4D
				);
		
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
	
	public void addGraphicInterface(Pane pane) {
		
	}



	public Double getTheta() {
		return theta;
	}



	public void setTheta(Double theta) {
		this.theta = theta;
	}






}
