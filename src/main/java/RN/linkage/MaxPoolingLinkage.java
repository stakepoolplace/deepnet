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
					
			subArea.applyMaxPoolingFilter(this, filterWidth, stride, (IPixelNode) thisNode, sigmaWI);
		
		}
			
		return sigmaWI.value();
	}
	
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {

	}
	
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		
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
