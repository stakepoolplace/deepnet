package RN.nodes;

import RN.algoactivations.EActivation;
import RN.links.Link;

/**
 *  Compute the product of their inputs .
 *  
 * @author Eric
 *
 */
public class SigmaPiNode extends Node {

	public SigmaPiNode() {
		super();
		this.nodeType = ENodeType.SIGMAPI;
	}
	
	public SigmaPiNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.SIGMAPI;
	}

	public SigmaPiNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.SIGMAPI;
	}

	@Override
	public double computeOutput(boolean trainingActive) throws Exception {
		double piI = 1D;

		// produit des entrees
		for (Link input : inputs) {
			piI *= input.getValue() * input.getWeight();
		}

		return piI;
	}

}
