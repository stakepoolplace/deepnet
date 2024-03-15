package RN.linkage;

import RN.AreaSquare;
import RN.IArea;
import RN.ILayer;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public class OneToOneFetchAreaLinkage extends Linkage {

	
	
	public OneToOneFetchAreaLinkage() {
	}

	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		// somme des entrees pondérées
		ILayer previouslayer = thisNode.getArea().getPreviousLayer();
		if(previouslayer != null){
			
			
			INode sourceNode = null;
			for(IArea area : getLinkedAreas()){
				
				if(area instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare){
					sourceNode = (INode) ((AreaSquare) area).getNodeXY(((IPixelNode)thisNode).getX(), ((IPixelNode)thisNode).getY(), sampling);
					sigmaWI += sourceNode.getComputedOutput();
				}else{
					sourceNode = area.getNode(thisNode.getNodeId());
					sigmaWI += sourceNode.getComputedOutput();
				}
				
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
	public void sublayerFanOutLinkage(INode thisNode, ILayer previouslayer) {
		
		if(previouslayer != null){
			
			try{
					
				for(IArea area : getLinkedAreas()){
					
					if(area instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare ){
						((AreaSquare) area).getNodeXY(((IPixelNode)thisNode).getX(), ((IPixelNode)thisNode).getY(), sampling).link(thisNode, ELinkType.REGULAR, false, 1D);
					}else{
						area.getNode(thisNode.getNodeId()).link(thisNode, ELinkType.REGULAR, false, 1D);
					}
				}
				
			}catch(Exception e){
				System.err.println("Unable to fan out one to one linkage from node :" + thisNode.getIdentification());
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
