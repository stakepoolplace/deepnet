package RN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import RN.algoactivations.EActivation;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.RecurrentNode;

/**
 * @author Eric Marchand
 * 
 */
public class Layer extends NetworkElement implements ILayer{

	protected int layerId;
	
	private EActivation function;
	
	List<ILayer> sublayers;

	List<IArea> areas;
	
	private boolean recurrent = false;

	private Boolean dropOutNodes = false;
	
	private double layerError = 0.0D;
	

	
	
	public Layer() {
		this.sublayers = new ArrayList<ILayer>();
		this.areas = new ArrayList<IArea>();
	}
	
	public Layer(EActivation eFunction) {
		this.function = eFunction;
		this.sublayers = new ArrayList<ILayer>();
		this.areas = new ArrayList<IArea>();
	}
	

	public Layer(Area... areas) {
		this();
		for(Area area : areas)
			addArea(area);
	}
	
	
	
	/* (non-Javadoc)
	 * @see RN.ILayer#isLayerReccurent()
	 */
	@Override
	public boolean isLayerReccurent(){
		return recurrent;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getAreaCount()
	 */
	@Override
	public int getAreaCount() {
		return areas.size();
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getNodeCount()
	 */
	@Override
	public int getNodeCount() {
		int count = 0;
		for (IArea area : areas) {
			count += area.getNodeCount();
		}
		return count;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#getNodeCountMinusRecurrentOnes()
	 */
	@Override
	public int getNodeCountMinusRecurrentOnes() {
		int count = 0;
		for (IArea area : areas) {
			count += area.getNodeCountMinusRecurrentOnes();
		}
		return count;
	}
	

	/* (non-Javadoc)
	 * @see RN.ILayer#getLayerId()
	 */
	@Override
	public Integer getLayerId() {
		return layerId;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setLayerId(int)
	 */
	@Override
	public void setLayerId(int layerId) {
		this.layerId = layerId;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#addArea(RN.IArea)
	 */
	@Override
	public void addArea(IArea area) {
		if(area.getLayer() == null){
			area.setAreaId(areas.size());
			area.setLayer(this);
		}
		areas.add(area);
		area.initGraphics();
	}
	
	
	/* (non-Javadoc)
	 * @see RN.ILayer#addAreas(RN.IArea)
	 */
	@Override
	public void addAreas(IArea... areas) {
		for(IArea area : areas){
			addArea(area);
		}
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getAreas()
	 */
	@Override
	public List<IArea> getAreas() {
		return areas;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#getArea(int)
	 */
	@Override
	public IArea getArea(int index) {
		return areas.get(index);
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setAreas(java.util.List)
	 */
	@Override
	public void setAreas(List<IArea> areas) {
		this.areas = areas;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getLayerNodes()
	 */
	@Override
	public List<INode> getLayerNodes() {
		List<INode> layerNodes = new ArrayList<INode>();
		List<IArea> areas = this.getAreas();
		for (IArea area : areas) {
			List<INode> nodes = area.getNodes();
			for (INode node : nodes) {
				layerNodes.add(node);
			}

		}
		return layerNodes;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#getLayerNodes(RN.nodes.ENodeType)
	 */
	@Override
	public List<INode> getLayerNodes(ENodeType... nodeType) {
		List<INode> layerNodes = new ArrayList<INode>();
		List<IArea> areas = this.getAreas();
		for (IArea area : areas) {
			
			List<INode> nodes = area.getNodes();
			
			for (INode node : nodes) {
				if(node.getNodeType() == ENodeType.ALL)
					layerNodes.add(node);
				else if(Arrays.asList(nodeType).contains(node.getNodeType()) )
					layerNodes.add(node);

			}

		}
		return layerNodes;
	}
	
	
	/* (non-Javadoc)
	 * @see RN.ILayer#toString()
	 */
	@Override
	public String toString(){
		return ITester.NEWLINE + ITester.NEWLINE +"Layer id : " + this.layerId;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getNetwork()
	 */
	@Override
	public Network getNetwork() {
		return network;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setNetwork(RN.Network)
	 */
	@Override
	public void setNetwork(Network network) {
		this.network = network;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#isFirstLayer()
	 */
	@Override
	public boolean isFirstLayer(){
		return this.getLayerId() == 0;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#isLastLayer()
	 */
	@Override
	public boolean isLastLayer(){
		return this.getLayerId() == network.getLayers().size() - 1;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#propagate(boolean)
	 */
	@Override
	public Double[] propagate(boolean playAgain) throws Exception {
		
		Double[] outputValues = null;
		
		for (IArea area : areas) {
			
			area.prePropagation();
			outputValues = area.propagation(playAgain, outputValues);
			area.postPropagation();
			
			if(area.isShowImage()){
				area.showImageArea();
			}
		}
		
		
		return outputValues;
		
	}


	/* (non-Javadoc)
	 * @see RN.ILayer#getFunction()
	 */
	@Override
	public EActivation getFunction() {
		return function;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setFunction(RN.algoactivations.EActivation)
	 */
	@Override
	public void setFunction(EActivation function) {
		this.function = function;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#finalizeConnections()
	 */
	@Override
	public void finalizeConnections() {
		
		if (network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED) {
			
			if (isFirstLayer()) {
				for (ILayer layer : getNetwork().getLayers()) {
					if (layer.isLayerReccurent()) {
						for (INode sourceNode : layer.getLayerNodes()) {
							this.getArea(0).addNode(new RecurrentNode(sourceNode));
						}
					}
				}
			}
			
		}

	}
	


	/* (non-Javadoc)
	 * @see RN.ILayer#setReccurent(boolean)
	 */
	@Override
	public void setReccurent(boolean b) {
		this.recurrent = b;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setDropOut(java.lang.Boolean)
	 */
	@Override
	public void setDropOut(Boolean dropOut) {
		dropOutNodes  = dropOut;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#isDropOutLayer()
	 */
	@Override
	public Boolean isDropOutLayer() {
		return this.dropOutNodes;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#getLayerError()
	 */
	@Override
	public double getLayerError() {
		return layerError;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#setLayerError(double)
	 */
	@Override
	public void setLayerError(double layerError) {
		this.layerError = layerError;
	}

	/* (non-Javadoc)
	 * @see RN.ILayer#deepCopy()
	 */
	@Override
	public Layer deepCopy() {
		Layer copy_layer = new Layer();
		
		List<IArea> copy_areas = new ArrayList<IArea>(areas);
		Collections.copy(copy_areas, areas);
		
		copy_layer.setDropOut(new Boolean(dropOutNodes));
		copy_layer.setFunction(EActivation.valueOf(function.name()));
		copy_layer.setLayerError(layerError);
		copy_layer.setLayerId(layerId);
		copy_layer.setReccurent(recurrent);
		copy_layer.setNetwork(network);
		copy_layer.setAreas(copy_areas);
		
		int idx = 0;
		for(IArea area : areas){
			area.setLayer(copy_layer);
			copy_areas.set(idx++, area.deepCopy());
		}
		
		
		return copy_layer;
	}
	
	/* (non-Javadoc)
	 * @see RN.ILayer#initGraphics()
	 */
	@Override
	public void initGraphics() {
		if(Graphics3D.graphics3DActive)
			Graphics3D.createLayer(this);
	}


}
