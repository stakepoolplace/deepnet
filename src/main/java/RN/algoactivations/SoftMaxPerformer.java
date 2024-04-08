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
	

    public SoftMaxPerformer(IArea area) {
    	this.area = area;
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
        
        return 1;
        
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

    	
        double derivative = 0.0; //  dérivée par rapport à chaque entrée.
       
        double max = 0;
        int maxIndex = 0;
        double[] all = new double[area.getNodes().size()];
        int idx = 0;
        for(INode nodeI : area.getNodes()) {
        	if(nodeI.getComputedOutput() > max) {
        		maxIndex = nodeI.getNodeId();
        	}
        	all[idx++] = nodeI.getComputedOutput();
        }
        
        for(INode nodeI : area.getNodes()) {
        	double yi = softmax(nodeI.getComputedOutput(), all);
        	
        	if(nodeI.getNodeId() != maxIndex) {
        		derivative = yi * (1 - yi);
        	} else {
        		derivative = -yi * yi;
        	}
        	nodeI.setDerivativeValue(derivative);
        }
        
        
        
        
//        for(INode nodeI : area.getNodes()) {
//        	
//	        int i = nodeI.getNodeId();
//        	double expSourceValue = 0;
//        	double backupedValue = 0;
//        	double sumExp = 0;
//        	for(Link link : nodeI.getInputs()) {
//        		INode sourceNode = link.getSourceNode();
//            	int j = sourceNode.getNodeId();
//        		expSourceValue = Math.exp(sourceNode.getComputedOutput());
//        		if(i == j) {
//        			backupedValue = expSourceValue;
//        		}
//        		sumExp += expSourceValue;
//        	}
//	        
//	        		
//	        double yi = backupedValue / sumExp;
//	        
//        
//	        for(Link link : nodeI.getInputs()) {
//	        
//		        INode nodeJ = link.getSourceNode();
//		        int j = nodeJ.getNodeId();
//		        double yj = Math.exp(nodeJ.getComputedOutput()) / sumExp;
//
//		        if(i == j) {
//		        	
//					derivative += yi * (1 - yi); // Cas où i = j
//		
//		        } else {
//		        	
//					derivative += -yi * yj; // Cas où i != j
//		
//		        }
//		        
//	        }
//	        
//			derivative += yi * (1 - yi); // Cas où i = j
//
//	        nodeI.setDerivativeValue(derivative);
//        
//        }

    	return 0D;
    	
    }
}
