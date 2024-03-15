package RN.linkage.vision;

import RN.IAreaSquare;
import RN.ILayer;
import RN.linkage.OneToOneLinkage;
import RN.links.ELinkType;
import RN.nodes.ENodeType;
import RN.nodes.INode;

public class GanglionaryCellLinkage extends OneToOneLinkage{

	
	public GanglionaryCellLinkage() {
		super();
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.GanglionaryLinkage#sublayerFanOutLinkage(RN.Layer)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {

		/**
		 * 
		 * 
		 * 
		 *  
		 *  
		 * 
		 */
		
		IAreaSquare subArea = (IAreaSquare) sublayer.getAreas().get(thisNode.getArea().getAreaId());
		INode sourceNode = subArea.getNode(thisNode.getNodeId());
		
			if(thisNode.getNodeType() == ENodeType.GANGLIONARY_OFF && sourceNode.getNodeType() == ENodeType.BIPOLAR_S){
				
				sourceNode.link(thisNode, ELinkType.OFF, false, 1D);
				
			}else if(thisNode.getNodeType() == ENodeType.GANGLIONARY_ON && sourceNode.getNodeType() == ENodeType.BIPOLAR_L){
				
				sourceNode.link(thisNode, ELinkType.ON, false, 1D);
				
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
	
}
