package RN.linkage;

import RN.ILayer;
import RN.nodes.INode;

public class DefaultLinkage extends Linkage {

	public DefaultLinkage(){
		
	}
	
	@Override
	public void initParameters() {
		// TODO Auto-generated method stub

	}

	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		// TODO Auto-generated method stub

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
