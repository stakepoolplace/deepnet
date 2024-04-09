package RN.algoactivations;

import java.io.Serializable;

import RN.IArea;
import RN.links.Link;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class SoftMaxPerformer implements Serializable, IActivation{
	
	private IArea area;
	private Link link;

    public SoftMaxPerformer(IArea area) {
    	this.area = area;
	}
    
    public SoftMaxPerformer(Link link) {
    	this.link = link;
	}

	@Override
    public double perform(double... values) throws Exception {
        
      
        for(INode softmaxNode : area.getNodes()) {
        	double sumExp = 0;
        	double expSourceValue = 0;
        	double backupedValue = 0;
        	int i = softmaxNode.getNodeId();
        	for(Link link : softmaxNode.getInputs()) {
        		INode sourceNode = link.getSourceNode();
            	int j = sourceNode.getNodeId();
        		expSourceValue = Math.exp(sourceNode.getComputedOutput());
        		if(i == j) {
        			backupedValue = expSourceValue;
        		}
        		sumExp += expSourceValue;
        	}


        	softmaxNode.setComputedOutput(backupedValue / sumExp);
        	
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

    	
    	INode sourceNode = this.link.getSourceNode();
    	INode targetNode = this.link.getTargetNode();
    	
    	int targetNodeCount = targetNode.getArea().getNodeCount();
    	int sourceNodeCount = sourceNode.getArea().getNodeCount();
    	if(targetNodeCount != sourceNodeCount) {
        	throw new RuntimeException("Softmax can only be calculated with 2 layers with equal number of nodes.");
    	}
    	
    	
        
    	double[] all = new double[sourceNode.getArea().getNodes().size()];
        int idx = 0;

        for(INode nodeI : sourceNode.getArea().getNodes()) {
        	all[idx++] = nodeI.getComputedOutput();
        }
    	double yi = softmax(targetNode.getComputedOutput(), all);
    	double yj = softmax(sourceNode.getComputedOutput(), all);


    	if(sourceNode.getNodeId() == targetNode.getNodeId()) {
    		return yi * (1 - yi); // Cas où i = j
    	} else {
    		return -yi * yj; // Cas où i != j
    	}

    	
    }
}
