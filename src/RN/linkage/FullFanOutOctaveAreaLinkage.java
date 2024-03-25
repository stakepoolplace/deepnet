package RN.linkage;

import RN.IArea;
import RN.IAreaSquare;
import RN.ILayer;
import RN.linkage.vision.KeyPoint;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public class FullFanOutOctaveAreaLinkage extends Linkage implements ILinkage {

	
	
	public FullFanOutOctaveAreaLinkage() {
	}
	
	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		Double angleMin = null;
		Double delta = (2D * Math.PI) / 9D;
		
		for(IArea area : getLinkedAreas()){
			
			OneToOneOctaveAreaLinkage linkage = (OneToOneOctaveAreaLinkage) area.getLinkage();
			IPixelNode sourceNode = null;
			
			angleMin = thisNode.getArea().getAreaId() * delta;
			
			Double magnitudeSum = null;
			Link link = null;
			
			for(KeyPoint kp : linkage.getKeyPoints()){
				
				if( kp.getX().intValue() / 4 == ((IPixelNode) thisNode).getX() && kp.getY().intValue() / 4 == ((IPixelNode) thisNode).getY()){
					
					sourceNode = ((IAreaSquare) area).getNodeXY(kp.getX().intValue(), kp.getY().intValue());
					magnitudeSum = kp.getMagnitudeSumByThetaRange(angleMin, delta);
					
					if(Math.abs(magnitudeSum) > 0D){
						link = Linkage.getLinkAndPutIfAbsent(thisNode, (INode) sourceNode, isWeightModifiable());
						sigmaWI += magnitudeSum * link.getWeight();
					}
				}
			}
			
		}
		
		if(thisNode.getBiasWeightValue() != null)
			sigmaWI -= thisNode.getBiasWeightValue();
		
		return sigmaWI;
	}
	
	
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#sublayerFanOutLinkage(RN.Layer)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer previousLayer){
		
		for (INode node : getLinkedArea().getNodes()) {
			
				node.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
		}
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.ILinkage#sublayerFanOutLinkage(RN.Layer, long)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT){
		
		for (IArea area : sublayer.getAreas()) {
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
