package RN.linkage;

import RN.AreaSquare;
import RN.IArea;
import RN.ILayer;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public class OneToOneFilterLinkage extends Linkage {

	
	
	public OneToOneFilterLinkage() {
	}

	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
			
		IArea subArea = getLinkedArea();
		INode sourceNode = null;
		
		if(subArea instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare){
			
			sourceNode = (INode) ((AreaSquare) subArea).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling);
			//sigmaWI +=  thisNode.getArea().getLayer().getLinkAndPutIfAbsent(thisNode, sourceNode, Link.getInstance(ELinkType.REGULAR, isWeightModifiable())).getWeight();
			
			// GAUSSIENNE - NEGATIF de l'image
			sigmaWI += sourceNode.getComputedOutput() - ((AreaSquare) network.getFirstLayer().getArea(0)).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling).getComputedOutput();
			
			//sigmaWI = Math.max(0D, sigmaWI);
			
		}else{
			sourceNode = subArea.getNode(thisNode.getNodeId());
			sigmaWI += sourceNode.getComputedOutput() * Linkage.getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
		}
			
		
		if(thisNode.getBiasWeightValue() != null)
			sigmaWI -= thisNode.getBiasWeightValue();
		
		return sigmaWI;
	}
	

	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
			
			IArea subArea = getLinkedArea();
			
			if(subArea instanceof AreaSquare && thisNode.getArea() instanceof AreaSquare){
				((AreaSquare) subArea).getNodeXY(((IPixelNode)thisNode).getX(), ((IPixelNode)thisNode).getY(), sampling).link(thisNode, ELinkType.REGULAR, false, 1D);
				((AreaSquare) network.getFirstLayer().getArea(0)).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling).link(thisNode, ELinkType.REGULAR, false, -1D);
			}else{
				subArea.getNode(thisNode.getNodeId()).link(thisNode, ELinkType.REGULAR, false, 1D);
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
