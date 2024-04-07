package RN.algoactivations;

import java.io.Serializable;

import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class SoftMaxPerformer implements Serializable, IActivation{
	
	private INode node;
	

    public SoftMaxPerformer(INode node) {
    	this.node = node;
	}

	@Override
    public double perform(double... values) throws Exception {
        

        double sumExp = 0;
        
        
//        for(INode currentNode : node.getArea().getLinkage().getIncomingNodes()) {
//        	sumExp += Math.exp(currentNode.getComputedOutput());
//        }
//
//        // Application de la fonction softmax à chaque valeur
//        return Math.exp(node.getInput(node.getNodeId()).getValue()) / sumExp;
        
        for(INode sameLayerNode : node.getArea().getNodes()) {
        	sumExp = Math.exp(sameLayerNode.getArea().getLinkage().getSigmaPotentials(sameLayerNode));
        	
        }

        
        return Math.exp(values[0]) / sumExp;
        
    }

    @Override
    public double performDerivative(double... values) throws Exception {

    	
    	
//        double derivativeSum = 0.0; // Ce sera la somme des dérivées par rapport à chaque entrée.
//        double yi = perform(values);
//        
//        i = node.getNodeId();
//        j = 
//        
//        if(i = j) {
//        	
//			derivativeSum = yi * (1 - yi); // Cas où i = j
//
//        } else {
//			derivativeSum = -yi * yj; // Cas où i != j
//
//        }
//        
//        
//        for(INode currentNode : node.getArea().getNodes()) {
//        	
//    		if(currentNode.getNodeId() == node.getNodeId()) {
//    			derivativeSum += yi * (1 - yi); // Cas où i = j
//    		} else {
//        		double sigmaWI = currentNode.getArea().getLinkage().getSigmaPotentials(currentNode);
//        		double yj = Math.exp(currentNode.getInput(currentNode.getNodeId()).getValue()) / sigmaWI;
//    			derivativeSum += -yi * yj; // Cas où i != j
//    		}
//
//        }
//        
//        return derivativeSum;  
    	
    	return 1D;
    	
    	
    }
}
