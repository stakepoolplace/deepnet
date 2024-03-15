package RN.linkage;

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
public class MaxPoolingLinkage extends FilterLinkage {

	
	Integer stride = null;
	Integer filterWidth = null;
	
	
	//private static Double[][] staticFilter = null;
	

	public MaxPoolingLinkage() {
	}
	
	public MaxPoolingLinkage(Integer sampling) {
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
		
//		staticFilter = new Double[filterWidth][filterWidth];
//		
//		for(int y=0; y < filterWidth; y++){
//			for(int x=0; x < filterWidth; x++){
//				staticFilter[x][y] = 1D;
//			}
//		}
		
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.Linkage#getLinkedSigmaPotentials(RN.nodes.INode)
	 */
	@Override
	public double getLinkedSigmaPotentials(INode thisNode){
		
		// Max Pooling depends on the outputs' computed of the previous layer
		// We can't links the nodes for the maxpooling process during the 'finalize network connections' phase, 
		// because previous layer outputs' are not yet computed.
		// So the linking is done during the feed-forward propagation.
		IAreaSquare subArea = null;
		for(IArea area : getLinkedAreas()){
			 subArea = (IAreaSquare) area;
		
			//initFilter(this, ID_FILTER_MAX_POOLING, null, (IPixelNode) thisNode, subArea);
			
			subArea.applyMaxPoolingFilter(this, filterWidth, stride, (IPixelNode) thisNode);
		
		}
			
		return super.getLinkedSigmaPotentials(thisNode);
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// Max Pooling depends on the outputs' computed of the previous layer
		// We can't get the max value for the maxpooling process during the 'finalize network connections' phase, 
		// because previous layer outputs' are not yet computed.
		// So this is done during the feed-forward propagation.
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = null;
		for(IArea area : getLinkedAreas()){
			 subArea = (IAreaSquare) area;
		
			//initFilter(this, ID_FILTER_MAX_POOLING, null, (IPixelNode) thisNode, subArea);
			
			subArea.applyMaxPoolingFilter(this, filterWidth, stride, (IPixelNode) thisNode, sigmaWI);
		
		}
			
		return sigmaWI.value();
	}
	
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		//IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		//initFilter(this, ID_FILTER_MAX_POOLING, null, (IPixelNode) thisNode, subArea);
		
		//subArea.applyMaxPoolingFilter(this, filterWidth, stride, (IPixelNode) thisNode);
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
	
}
