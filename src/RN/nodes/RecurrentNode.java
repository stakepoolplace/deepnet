package RN.nodes;

/**
 * @author Eric Marchand
 *
 */
public class RecurrentNode extends Node {
	
	
	Integer sourceLayerId = null;
	INode sourceNode = null;
	
	
	public RecurrentNode(INode sourceNode) {
		super();
		this.sourceNode = sourceNode;
		this.nodeType = ENodeType.RECURRENT;
	}

	public Integer getSourceLayerId() {
		return sourceLayerId;
	}

	public void setSourceLayerId(Integer sourceLayerId) {
		this.sourceLayerId = sourceLayerId;
	}

	public INode getSourceNode() {
		return sourceNode;
	}

	public void setSourceNode(Node sourceNode) {
		this.sourceNode = sourceNode;
	}
	
	public String toString() {
		return "Recurrent" + super.toString();
	}



}
