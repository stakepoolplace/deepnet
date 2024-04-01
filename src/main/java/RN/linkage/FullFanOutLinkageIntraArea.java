package RN.linkage;

import RN.IArea;
import RN.ILayer;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class FullFanOutLinkageIntraArea extends Linkage implements ILinkage {

	
	
	public FullFanOutLinkageIntraArea() {
	}
	
	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		IArea subArea = getLinkedArea();
		for (INode sourceNode : subArea.getNodes()) {
			//if(getContext().getClock() == -1 || input.getFireTimeT() == getContext().getClock()){
				sigmaWI += sourceNode.getComputedOutput() * getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
				//input.synchFutureFire();
			//}
			
		}
			
		sigmaWI -= thisNode.getBiasWeightValue();
			
		
		
		return sigmaWI;
	}
	
	
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#sublayerFanOutLinkage(RN.Layer)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer previousLayer){
		
		IArea subArea = getLinkedArea();
		for (INode node : subArea.getNodes()) {
			
				node.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
		}
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#sublayerFanOutLinkage(RN.Layer, long)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT){
		
		for(IArea area : sublayer.getAreas()){
		for (INode node : area.getNodes()) {
			
				node.link(thisNode, ELinkType.REGULAR, isWeightModifiable()).setFireTimeT(initFireTimeT);
		}
		}
		
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#nextLayerFanOutLinkage(RN.Layer)
	 */
	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer){
		
		IArea nextArea = nextlayer.getArea(thisNode.getArea().getAreaId());
		
		for (INode node : nextArea.getNodes()) {
				Link link = thisNode.link(node, ELinkType.REGULAR, isWeightModifiable());
				node.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
				node.getInputs().add(link);
		}
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#nextLayerFanOutLinkage(RN.Layer, long)
	 */
	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT){
		
		IArea nextArea = nextlayer.getArea(thisNode.getArea().getAreaId());
		
		for (INode node : nextArea.getNodes()) {
				Link link = thisNode.link(node, ELinkType.REGULAR, isWeightModifiable());
				node.link(thisNode, ELinkType.REGULAR, isWeightModifiable()).setFireTimeT(initFireTimeT);
				node.getInputs().add(link);
		}
		
	}

	@Override
	public void initParameters() {
		// TODO Auto-generated method stub
		
	}



	
	
	
}
