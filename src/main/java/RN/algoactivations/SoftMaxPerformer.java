package RN.algoactivations;

import java.io.Serializable;

import RN.IArea;
import RN.algoactivations.utils.BoundMath;
import RN.links.Link;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class SoftMaxPerformer implements Serializable, IActivation{
	
	private IArea area;
	
    public SoftMaxPerformer() {
	}

    public SoftMaxPerformer(IArea area) {
    	this.area = area;
	}

	@Override
    public double perform(double... values) throws Exception {
		
		double[] logits = new double[area.getNodes().size()];
		double[] max = new double[area.getNodes().size()];
		double sumExp = 0D;
		for(INode softmaxNode : area.getNodes()) {
			logits[softmaxNode.getNodeId()] = 0D;
			max[softmaxNode.getNodeId()] = Double.NEGATIVE_INFINITY;
			
			// avoid bias
			//logits[softmaxNode.getNodeId()] = area.getLinkage().getLinkedSigmaPotentialsUnsync(softmaxNode);
        	for(Link link : softmaxNode.getInputs()) {
        		double weightedValue = link.getValue() * link.getWeight();
        		logits[softmaxNode.getNodeId()] += weightedValue;
        		if(weightedValue > max[softmaxNode.getNodeId()]) {
        			max[softmaxNode.getNodeId()] = weightedValue;
        		}
        	}

    		sumExp += BoundMath.exp(logits[softmaxNode.getNodeId()] - max[softmaxNode.getNodeId()]);

		}
		
		for(INode softmaxNode : area.getNodes()) {
			
        	softmaxNode.setComputedOutput(BoundMath.exp(logits[softmaxNode.getNodeId()] - max[softmaxNode.getNodeId()]) / sumExp);
        	softmaxNode.setError(softmaxNode.getIdealOutput() - softmaxNode.getComputedOutput());
        	softmaxNode.setDerivativeValue(1D);
        	
        	if(softmaxNode.getComputedOutput().isNaN())
        		throw new RuntimeException("maybe in UNLINKED net ... " + sumExp );
        	
		}
        
        
        return 0;
        
    }
	


    @Override
    public double performDerivative(double... values) throws Exception {

    	
    	// useless since dCdz[i] = t[i] - a[i];
    	
    	return 1D;

    	
    }
}
