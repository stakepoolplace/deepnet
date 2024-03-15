package RN.linkage;

import java.util.ArrayList;
import java.util.List;

import RN.IArea;
import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author ericmarchand
 * 
 */
public class MaxLinkage extends FilterLinkage {

	
	Integer stride = null;
	Integer filterWidth = null;
	Integer halfWidth = null;
	
	private static List<IPixelNode> maxMinNodes = new ArrayList<IPixelNode>();
	
	
	//private static Double[][] staticFilter = null;
	

	public MaxLinkage() {
	}
	
	public MaxLinkage(Integer sampling) {
		this.sampling = sampling;
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#initParameters()
	 */
	public void initParameters() {
		
		if(params[0] != null)
			filterWidth = params[0].intValue();
		
		if(params[1] != null)
			stride = params[1].intValue();
		
		halfWidth = ((filterWidth - 1) / 2);
		
//		staticFilter = new Double[filterWidth][filterWidth];
//		
//		for(int y=0; y < filterWidth; y++){
//			for(int x=0; x < filterWidth; x++){
//				staticFilter[x][y] = 1D;
//			}
//		}
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		//SigmaWi sigmaWI = new SigmaWi();
		IPixelNode pix = (IPixelNode) thisNode;
		
		
		IAreaSquare subArea = null;
		IPixelNode centerPix = null;

		List<IPixelNode> nodesInSquare = null;
		
		boolean isMinMax = true;
		
		int x = pix.getX() * stride;
		int y = pix.getY() * stride;
		int x0 = x - halfWidth;
		int y0 = y - halfWidth;
		
		for(IArea area : getLinkedAreas()){
			
			isMinMax = true;
			
			 subArea = (IAreaSquare) area;
			 centerPix = subArea.getNodeXY(x, y);
			 
			 if(isMinMax && centerPix.getPreviousAreaSquare() != null){
				 nodesInSquare = centerPix.getPreviousAreaSquare().getNodesInSquareZone(x0, y0, filterWidth, filterWidth);
				 isMinMax &= isMinMax(centerPix, nodesInSquare);
			 }
			 
			 if(isMinMax){
				 nodesInSquare = subArea.getNodesInSquareZone(x0, y0, filterWidth, filterWidth);
				 isMinMax &= isMinMax(centerPix, nodesInSquare);
			 }
			 
			 if(isMinMax && centerPix.getNextAreaSquare() != null){
				 nodesInSquare = centerPix.getNextAreaSquare().getNodesInSquareZone(x0, y0, filterWidth, filterWidth);
				 isMinMax &= isMinMax(centerPix, nodesInSquare);
			 }
			 
			 if(isMinMax){
				 maxMinNodes.add(pix);
				 //return 1D;
				 return centerPix.getComputedOutput();
			 }
			
		}
			
		return  0D;
	}
	
	private boolean isMinMax(IPixelNode centerPix, List<IPixelNode> nodesInSquare){
		
		boolean isMin = true;
		boolean isMax = true;
		
		for(IPixelNode node : nodesInSquare){
			
			if(node == centerPix)
				continue;
			
			isMin &= centerPix.compareOutputTo(node) < 0D;
			isMax &= centerPix.compareOutputTo(node) > 0D;
				
			if(!isMin && !isMax)
				return false;
		}
		
		return true;
		
	}
	
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		//initFilter(this, ID_FILTER_MAX_POOLING, null, (IPixelNode) thisNode, subArea);
		
		subArea.applyMaxPoolingFilter(this, filterWidth, stride, (IPixelNode) thisNode);
	}
	
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
//		int x = sublayerNode.getX();
//		int y = sublayerNode.getY();
//		
//		if(x >= 0 && x <= (filterWidth-1) && y >= 0 && y <= (filterWidth-1)){
//			return staticFilter[x][y];
//		}
		
		return 0D;
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

	public static List<IPixelNode> getMaxMinNodes() {
		return maxMinNodes;
	}

	public static void setMaxMinNodes(List<IPixelNode> maxMinNodes) {
		MaxLinkage.maxMinNodes = maxMinNodes;
	}
	
}
