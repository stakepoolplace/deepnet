package RN.linkage;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import RN.IArea;
import RN.IAreaSquare;
import RN.ILayer;
import RN.Identification;
import RN.dataset.inputsamples.ESamples;
import RN.links.Link;
import RN.links.Weight;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author ericmarchand
 * 
 */
public class SubsamplingLinkage extends FilterLinkage {

	
	private Integer stride = null;
	private Integer filterWidth = null;
	private Integer toCenter = null;
	private static Map<Identification,Weight> trainableCoefs = null;
	private static Map<Identification,Weight> sharedBiasWeight = null;

	public SubsamplingLinkage() {
	}
	
	public SubsamplingLinkage(Integer sampling) {
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
		
		toCenter = ((filterWidth - 1) / 2);
		
		trainableCoefs = null;
		sharedBiasWeight = null;
	}
	
	private void initTrainableCoefs(){
		
		if (trainableCoefs == null || trainableCoefs.get(getArea().getIdentification()) == null) {
			trainableCoefs = new HashMap<Identification, Weight>();
			sharedBiasWeight = new HashMap<Identification, Weight>();
			
			Weight biasWeight = null;
			for (IArea area : getArea().getLayer().getAreas()) {
				biasWeight = new Weight();
				trainableCoefs.put(area.getIdentification(), new Weight());
				sharedBiasWeight.put(area.getIdentification(), biasWeight);
				for(INode node : area.getNodes()){
					node.setBiasWeight(biasWeight);
				}
			}
		}
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		IPixelNode pix = (IPixelNode) thisNode;
		
		IAreaSquare subArea = null;
		IPixelNode centerPix = null;

		List<IPixelNode> nodesInSquare = null;
		
		initTrainableCoefs();
		
		for(IArea area : getLinkedAreas()){
			
			 subArea = (IAreaSquare) area;
			 centerPix = subArea.getNodeXY(pix.getX() * stride , pix.getY() * stride);
			 
			nodesInSquare = subArea.getNodesInSquareZone(centerPix.getX() - toCenter, centerPix.getY() - toCenter, filterWidth, filterWidth);
			for(IPixelNode innerPix : nodesInSquare){
				sigmaWI.sum(innerPix.getComputedOutput());
			}
			
			sigmaWI.multiply(getLinkAndPutIfAbsent(thisNode, (INode) centerPix, isWeightModifiable(), trainableCoefs.get(getArea())).getWeight());
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
		
		initTrainableCoefs();
		
		for(IArea area : getLinkedAreas()){
			
			 subArea = (IAreaSquare) area;
			 centerPix = subArea.getNodeXY(pix.getX() * stride , pix.getY() * stride);
			 
			nodesInSquare = subArea.getNodesInSquareZone(centerPix.getX() - toCenter, centerPix.getY() - toCenter, filterWidth, filterWidth);
			for(IPixelNode innerPix : nodesInSquare){
				innerPix.link(thisNode, isWeightModifiable(), trainableCoefs.get(getArea()));
			}
			
		}
			
	}
	
	public double getLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		// somme des entrees pondérées
		for (Link input : thisNode.getInputs()) {
			
			if(getContext().getClock() == -1 || input.getFireTimeT() == getContext().getClock()){
				
				sigmaWI += input.getValue();
				
				input.synchFutureFire();
			}
			
		}
		
		
		// ajout du biais
		if (thisNode.getBiasInput() != null){
			if(getContext().getClock() == -1 || thisNode.getBiasInput().getFireTimeT() == getContext().getClock()){
				
				sigmaWI -= thisNode.getBiasInput().getValue() * thisNode.getBiasInput().getWeight();
				
				thisNode.getBiasInput().synchFutureFire();
			}
		}
		
		return sigmaWI;
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
