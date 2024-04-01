package RN.parallel;

import java.util.List;

import RN.ILayer;
import RN.links.Link;
import RN.nodes.INode;

public class PropagationProblem {

	public static ILayer layer;
	public int idxStart;
	public int idxStop;
	

	public PropagationProblem(int idxStart, int idxStop) {
		this.idxStart = idxStart;
		this.idxStop = idxStop;
	}
	

	public void propagate() {
		 if(idxStop > layer.getNodeCount() - 1)
			 idxStop = layer.getNodeCount() - 1;
		 propagate(layer.getLayerNodes().subList(idxStart, idxStop));
	}
	
	private void propagate(List<INode> nodes) {
		for(INode node : nodes){
			propagate(node);
		}
		
	}

	private Double propagate(INode node) {

		System.out.println("Thread: " + Thread.currentThread().getName() + " calculates " + node);
		
		try {
			node.computeOutput(false);
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (node.getInputs().size() == 1 && node.getInput(0).getSourceNode() == null) {

			return node.getComputedOutput();

		} else {

			for (Link currentLink : node.getInputs()) {
				propagate(currentLink.getSourceNode());
			}

			return node.getComputedOutput();

		}
	}

}
