package RN;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import RN.algoactivations.EActivation;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.utils.ReflexionUtils;

/**
 * @author Eric Marchand
 * 
 */
public class NodeFactory implements Serializable {
	
	public NodeFactory(){
	}
	
	
	private boolean bias = false;
	
	private EActivation activation = null;
	
	private ENodeType[] types = null;

	
	
	
	public void configureNode(boolean bias, EActivation activation, ENodeType... types) {
		this.bias = bias;
		this.activation = activation;
		this.types = types;
	}
	
	public void configureNode(boolean bias, ENodeType... types){
		this.bias = bias;
		this.types = types;
	}
	
	
	public List<INode> createNodes(IArea area, int nodeCount){
		
		List<INode> nodes = new ArrayList<INode>(nodeCount);
		
		for (int ind = 1; ind <= nodeCount; ind++) {
			
			INode obj = null;
			
			for(ENodeType type : types){
				obj = instantiateNeuron(ind, activation, type, obj);
			}
			
			switch(types[types.length - 1]){
				case LSTM:
					addNode(obj, area, nodes);
				break;
					
				case PIXEL:
				case GANGLIONARY_OFF:
				case GANGLIONARY_ON:
				case BIPOLAR_L:
				case BIPOLAR_S:
					addNode(obj, area, nodes);
					((INode) obj).initParameters(nodeCount);
				break;
					
				case REGULAR:
				case IMAGE :
				case RECURRENT:
				case TIMESERIE:
				case ALL:
				default :
					addNode(obj, area, nodes);
			}

			if(area.getLinkage() == null)
				throw new RuntimeException("You must define linkage on area before creating nodes.");
			
			
			if(bias)
				((INode) obj).createBias();
			
		}
		
		return nodes;
	}
	
	private void addNode(INode node, IArea area, List<INode> nodes){
		
		node.setArea(area);
		node.setNodeId(area.getNodes() != null && !area.getNodes().isEmpty() ? area.getNodes().size() + nodes.size() : nodes.size());
		nodes.add(node);
		node.initGraphics();
	}
	
	

	
	private <T> T instantiateNeuron(int idxNode, EActivation activation, ENodeType type, T obj) {
		
		Class[] constructorParamClasses = new Class[]{};
		Object[] constructorParamObject = new Object[]{};

		
		// on affiche qu'un visualiseur
		if(idxNode > 1 && type.equals(ENodeType.IMAGE) )
			return null;
			
		if(obj == null){
			
			if(activation != null){
				constructorParamClasses = new Class[]{EActivation.class};
				constructorParamObject = new Object[]{activation};
			}
			
			obj = ReflexionUtils.newClass(type.getClassName(), constructorParamClasses, constructorParamObject);
		
		}else{
			
			if(activation != null){
				constructorParamClasses = new Class[]{EActivation.class, INode.class};
				constructorParamObject = new Object[]{activation, obj};
			}else{
				constructorParamClasses = new Class[]{INode.class};
				constructorParamObject = new Object[]{obj};
			}
			
			obj = ReflexionUtils.newClass(type.getClassName(), constructorParamClasses, constructorParamObject);
		}
		return obj;
	}

	

}
