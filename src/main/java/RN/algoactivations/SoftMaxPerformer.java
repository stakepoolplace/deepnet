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
        
      
        for(INode softmaxNode : area.getNodes()) {
        	
        	Double sumExp = 0D;
        	Double expSourceValue = 0D;
        	Double backupedValue = 0D;
        	int i = softmaxNode.getNodeId();
        	
        	for(Link link : softmaxNode.getInputs()) {
        		INode sourceNode = link.getSourceNode();
            	int j = sourceNode.getNodeId();
        		expSourceValue = BoundMath.exp(sourceNode.getComputedOutput());
        		if(i == j) {
        			backupedValue = expSourceValue;
        		}
        		sumExp += expSourceValue;
        	}

        	softmaxNode.setComputedOutput(backupedValue / sumExp);
        	softmaxNode.setError(softmaxNode.getComputedOutput() - softmaxNode.getIdealOutput() );
        	softmaxNode.setDerivativeValue(1D);
        	
        	if(softmaxNode.getComputedOutput().isNaN())
        		throw new RuntimeException("maybe in UNLINKED net ... " + backupedValue + " " + sumExp );
        	
        	
        }
        
        return 0;
        
    }
	
	private double softmax(double val, double...values) {
		
		double sumExp = 0;
		for(double det: values) {
			sumExp += Math.exp(det);
		}
		
		return Math.exp(val) / sumExp;
	}

    @Override
    public double performDerivative(double... values) throws Exception {

    	
    	// useless since dCdz[i] = t[i] - a[i];
    	
    	return 1D;

    	
    }
}
