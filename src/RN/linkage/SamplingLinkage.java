package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public class SamplingLinkage extends FilterLinkage {

	
	public SamplingLinkage() {
	}
	
	public SamplingLinkage(Integer sampling) {
		this.sampling = sampling;
	}
	
	public void initParameters() {
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			
		if(sampling != null && sampling != 1){
			initFilter(this, ID_FILTER_SAMPLING_2, ESamples.SAMPLING, (IPixelNode) thisNode, subArea);
			subArea.applyFilter(this, ID_FILTER_SAMPLING_2, (IPixelNode) thisNode, sigmaWI);
		}
			
		
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		if(sampling != null && sampling != 1){
			initFilter(this, ID_FILTER_SAMPLING_2, ESamples.SAMPLING, (IPixelNode) thisNode, subArea);
			subArea.applyFilter(this, ID_FILTER_SAMPLING_2, (IPixelNode) thisNode, 0f);
		}
		
		
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
