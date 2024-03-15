package RN.linkage;

import RN.AreaSquare;
import RN.IArea;
import RN.ILayer;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public class OneToOneLinkage extends Linkage {

	
	
	public OneToOneLinkage() {
	}

	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		// somme des entrees pondérées
		IArea subArea = getLinkedArea();
		if(subArea != null){
			
			INode sourceNode = null;
			if(subArea instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare){
				
				sourceNode = (INode) ((AreaSquare) subArea).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling);
				sigmaWI += sourceNode.getComputedOutput() * getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
				
			}else{
				sourceNode = subArea.getNode(thisNode.getNodeId());
				sigmaWI += sourceNode.getComputedOutput() * getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
			}
			
		}else{
			//if(getContext().getClock() == -1 || input.getFireTimeT() == getContext().getClock()){
			sigmaWI = thisNode.getInputValue() ; //* input.getWeight();
				//input.synchFutureFire();
			//}
				
		}
		
		if(thisNode.getBiasWeightValue() != null)
			sigmaWI -= thisNode.getBiasWeightValue();
		
		return sigmaWI;
	}
	

	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IArea subArea = getLinkedArea();
		
		if(subArea != null){
			
			if(subArea instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare){
				((AreaSquare) subArea).getNodeXY(((IPixelNode)thisNode).getX(), ((IPixelNode)thisNode).getY(), sampling).link(thisNode, ELinkType.REGULAR, isWeightModifiable(), 1D);
			}else{
				subArea.getNode(thisNode.getNodeId()).link(thisNode, ELinkType.REGULAR, isWeightModifiable(), 1D);
			}
			
			
		}else{
			thisNode.incomingLink(ELinkType.REGULAR);
		}
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

	@Override
	public void initParameters() {
		// TODO Auto-generated method stub
		
	}




}
