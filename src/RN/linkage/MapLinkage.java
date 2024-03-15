package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public class MapLinkage extends FilterLinkage {

	Integer stride = null;
	Integer mapWidth = null;
	
	public MapLinkage() {
	}
	
	public MapLinkage(Integer sampling) {
		this.sampling = sampling;
	}
	
	public void initParameters() {
		
		if(params[0] != null)
			mapWidth = params[0].intValue();
		
		
		if(params[1] != null)
			stride = params[1].intValue();
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		IPixelNode pix = (IPixelNode) thisNode;
		
		Double max = null;
			
		for(IPixelNode subPix : subArea.getNodesInSquareZone(pix.getX() * stride, pix.getY() * stride, mapWidth, mapWidth)){
			max = (max == null ? subPix.getComputedOutput() : Math.max(subPix.getComputedOutput(), max)); 
		}
			
		
		return max == null ? 0D : max;
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		IPixelNode pix = (IPixelNode) thisNode;
		
		IPixelNode maxNode = null;
		
			
		for (IPixelNode subPix : subArea.getNodesInSquareZone(pix.getX() * stride, pix.getY() * stride, mapWidth, mapWidth)) {
			if(maxNode == null || subPix.getComputedOutput() > maxNode.getComputedOutput())
				maxNode = subPix;
		}

		if(maxNode != null)
			maxNode.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		// Calcul du filtre gaussien
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) sublayerNode.getX(),
				(double) sublayerNode.getY(), 
				(double) sampling
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


}
