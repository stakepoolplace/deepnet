package RN.nodes;

import RN.algoactivations.EActivation;
import RN.links.Link;

/**
 *  Compute the product of their inputs . These units have no weights.
 *  
 * @author Eric
 *
 */
public class ProductNode extends Node {

	public ProductNode() {
		super();
		this.nodeType = ENodeType.PRODUCT;
	}
	
	public ProductNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.PRODUCT;
	}

	public ProductNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.PRODUCT;
	}

	@Override
	public double computeOutput(boolean trainingActive) throws Exception {
		double piI = 1D;

		// produit des entrees
		for (Link input : inputs) {
			piI *= input.getValue();
		}
		
		if(innerNode != null)
			innerNode.showImage(this);

		return piI;
	}

}
