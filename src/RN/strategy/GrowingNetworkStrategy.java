package RN.strategy;

import RN.ILayer;
import RN.TestNetwork;
import RN.links.Link;
import RN.nodes.Node;

public class GrowingNetworkStrategy extends AbstractStrategyDecorator implements IStrategy{

	
	public GrowingNetworkStrategy() {
	}

	public GrowingNetworkStrategy(Strategy strategy){
		this.decoratedStrategy = strategy;
	}

	@Override
	public void beforeProcess() {
		
	}
	
	@Override
	public void process() {
		
		ILayer subLayer = network.getFirstLayer();
		ILayer growingLayer = network.getLayer(subLayer.getLayerId() + 1);
		ILayer nextLayer = network.getLayer(growingLayer.getLayerId() + 1);
		
		Node newNode = new Node();
		newNode.createBias();
		growingLayer.getArea(growingLayer.getAreaCount() - 1).addNode(newNode);
		
		if(network.getLayers().size() > 3){
			ILayer secondSubLayer = growingLayer;
			ILayer secondGrowingLayer = network.getLayer(secondSubLayer.getLayerId() + 2);
			ILayer secondNextLayer = network.getLayer(secondGrowingLayer.getLayerId() + 1);
			
			Node secondNewNode = new Node();
			secondNewNode.createBias();
			secondGrowingLayer.getArea(secondGrowingLayer.getAreaCount() - 1).addNode(secondNewNode);
			
			// we link after adding nodes
			crossLinkage(secondNewNode, secondSubLayer, secondGrowingLayer, secondNextLayer);
		}
		
		// we link after adding nodes
		crossLinkage(newNode, subLayer, growingLayer, nextLayer);
		
		
	}

	private void crossLinkage(Node newNode, ILayer subLayer, ILayer growingLayer, ILayer nextLayer) {
		TestNetwork tester = TestNetwork.getInstance();
		
		newNode.getArea().getLinkage().sublayerFanOutLinkage(newNode, subLayer);
		
		newNode.getArea().getLinkage().nextLayerFanOutLinkage(newNode, nextLayer);
		
		for (Link link : newNode.getInputs())
			link.initWeight(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
		
		if (growingLayer.isDropOutLayer()) {
			newNode.setDropOutActive(true);
		}
	}


	@Override
	public void afterProcess() {
	}
	
}
