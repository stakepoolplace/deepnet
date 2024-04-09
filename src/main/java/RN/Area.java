package RN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import RN.algoactivations.EActivation;
import RN.algoactivations.IActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.DefaultLinkage;
import RN.linkage.EFilterPosition;
import RN.linkage.ELinkage;
import RN.linkage.ELinkageBetweenAreas;
import RN.linkage.FilterLinkage.FilterIndex;
import RN.linkage.IFilterLinkage;
import RN.linkage.ILinkage;
import RN.linkage.SigmaWi;
import RN.links.ELinkType;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.ImageNode;
import RN.nodes.LSTMNode;
import RN.nodes.Node;
import RN.nodes.PixelNode;
import RN.nodes.RecurrentNode;
import RN.utils.ReflexionUtils;

/**
 * @author Eric Marchand
 * 
 */
public class Area extends NetworkElement implements IArea{

	protected int areaId = 0;
	
	protected List<INode> nodes;

	protected ILayer layer;
	
	protected Boolean showImage = Boolean.FALSE;
	
	protected ImageNode imageArea = null;
	
	protected Integer widthPx = null;
	
	protected Integer heightPx = null;

	protected Integer nodeCenterX = null;
	
	protected Integer nodeCenterY = null;
	
	protected String comment = "";
	
	protected NodeFactory nodeFactory = new NodeFactory();
	
	protected ILinkage linkage = new DefaultLinkage();
	
	protected EActivation activation = null;
	protected IActivation performer = null;
	


	Area(){
	}
	
	public Area(int nodeCount) {
	
		this.nodes = new ArrayList<INode>(nodeCount);
		
	}
	
	public Area(int nodeCount, boolean showImage) {
		
		this.nodes = new ArrayList<INode>(nodeCount);
		this.showImage = showImage;
		initWidthPx(nodeCount);
		if(showImage){
			this.imageArea = new ImageNode(EActivation.IDENTITY, this.widthPx, this.heightPx);
			this.imageArea.setArea(this);
		}

	}
	
	public Area(EActivation activation, int nodeCount) {
		
		this.nodes = new ArrayList<INode>(nodeCount);
		this.activation = activation;
		if(activation != null) {
			performer = EActivation.getPerformer(activation, this);
		}
		
	}
	
	public void initWidthPx(int pixSize){
		
		if(pixSize == 0D)
			throw new RuntimeException("Le nombre de neurones dans la liste est vide.");
		
		// Colonne
		this.widthPx = 1;
		this.heightPx = pixSize;
		this.nodeCenterX = 1;
		this.nodeCenterY = pixSize / 2;
	}
	
	public void showImageArea() {
		comment = ": " + this.widthPx + "x" + this.heightPx;
		imageArea.getStage().setTitle((layer != null ? "Layer #" + layer.getLayerId() : "") + " Area #" + areaId + " " + comment);
		imageArea.initImageData();
		imageArea.insertDataArea();
		imageArea.drawImageData(null);

	}
	

	public List<INode> createNodes(int nodeCount){
		
		if(this.nodes != null && !this.nodes.isEmpty())
			this.nodes.addAll(nodeFactory.createNodes(this, nodeCount));
		else
			this.nodes = nodeFactory.createNodes(this, nodeCount);
		
		return this.nodes;
		
	}
	
	public IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, Integer[] targetedAreas, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, linkageAreas, targetedAreas, eSampleFunction, linkageWeightModifiable, optParams);
		
		return this;
	}
	

	
	public IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, Integer[] targetedAreas, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable, Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, linkageAreas, targetedAreas, eSampleFunction, linkageWeightModifiable, optParams);
		this.linkage.setSampling(sampling);
		
		return this;
	}
	
	public IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable, Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, linkageAreas, null, eSampleFunction, linkageWeightModifiable, optParams);
		this.linkage.setSampling(sampling);
		
		return this;
	}
	
	public IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, linkageAreas, null, eSampleFunction, linkageWeightModifiable, optParams);
		
		return this;
	}
	
	public IArea configureLinkage(ELinkage linkage, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable,  Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, null, null, eSampleFunction, linkageWeightModifiable, optParams);
		this.linkage.setSampling(sampling);
		
		return this;
	}
	
	public IArea configureLinkage(ELinkage linkage, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams){
		
		this.linkage = instantiateLinkage(linkage, this, null, null, eSampleFunction, linkageWeightModifiable, optParams);
		
		return this;
	}
	
	public IArea configureNode(boolean bias, EActivation activation, ENodeType type){
		
		this.nodeFactory.configureNode(bias, activation,  type);
		
		return this;
		
	}
	
	public IArea configureNode(boolean bias, ENodeType... types){
		
		this.nodeFactory.configureNode(bias, types);
		
		return this;
		
	}
	
	
	private ILinkage instantiateLinkage(ELinkage linkageType, Area thisArea, ELinkageBetweenAreas linkageAreas, Integer[] targetedAreas, ESamples eSampleFunction, boolean weightModifiable, Double[] params) {
		
		ILinkage linkage = ReflexionUtils.newClass(linkageType.getClassPath(), new Class[]{}, new Object[]{});
		
		linkage.setArea(thisArea);
		
		linkage.setLinkageType(linkageType);
		
		if(linkageAreas != null){
			linkage.setLinkageAreas(linkageAreas);
		}
		
		if(targetedAreas != null){
			linkage.setTargetedAreas(Arrays.asList(targetedAreas));
		}
		
		linkage.setWeightModifiable(weightModifiable);
		
		if(params != null){
			linkage.setParams(params);
		}
		linkage.initParameters();
		
		if(eSampleFunction != null)
			((IFilterLinkage)linkage).setESampleFunction(eSampleFunction);
		
		return linkage;
	}
	
	
	public void initBiasWeights(double value) {

		
			List<INode> areaNodes = this.getNodes();
			for (INode node : areaNodes) {
				
				if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED)
					node.getBiasInput().initWeight(value);
				else
					node.setBiasWeightValue(value);
			}


	}
	
	
	public Double[] propagation(boolean playAgain, Double[] outputValues) throws Exception {
		
		List<INode> nodes = (List<INode>) this.getNodes();
		
		if (getLayer().isLastLayer()){
			outputValues = new Double[nodes.size()];
		}
		
		if(!playAgain || getLinkage().isWeightModifiable() || getLayer().isFirstLayer()){
			
			for (INode node : nodes) {
				
				node.computeOutput(playAgain);
				
				if (getLayer().isLastLayer()) {
					outputValues[node.getNodeId()] = node.getComputedOutput();
					node.setError(node.getIdealOutput() - node.getComputedOutput());
				}
			}
		}
		
		return outputValues;
	}
	
	

	/**
	 * 
	 */
	public void finalizeConnections() {
		
		
		if (network.getImpl() == ENetworkImplementation.LINKED) {
			
			INode lastRecurrent = null;
			INode lastTimeSerie = null;
			List<INode> nodes = (List<INode>) this.getNodes();
			
			for (INode node : nodes) {
				if (getLayer().getNetwork().isRecurrentNodesLinked() && node.getNodeType() == ENodeType.RECURRENT) {
					
					if (lastRecurrent != null)
						lastRecurrent.doubleLink((RecurrentNode) node, ELinkType.RECURRENT_LATERAL_LINK);
					
					lastRecurrent = node;
				}
				
				if(node.getNodeType() == ENodeType.TIMESERIE){
					if (lastTimeSerie != null)
						lastTimeSerie.link(node, ELinkType.RECURRENT_LATERAL_LINK);
					
					lastTimeSerie = node;
				}
			}
			
		
			LSTMNode lastLSTMNode = null;
			for (INode node : nodes) {
				if(node.getNodeType() == ENodeType.LSTM){
					LSTMNode currentLSTMNode = (LSTMNode) node;
					if (lastLSTMNode != null){
						for(Node memory : lastLSTMNode.getMemories()){
							memory.link(currentLSTMNode.getInputGate(), ELinkType.RECURRENT_LINK);
							memory.link(currentLSTMNode.getForgetGate(), ELinkType.RECURRENT_LINK);
							memory.link(currentLSTMNode.getOutputGate(), ELinkType.RECURRENT_LINK);
						}
						
						lastLSTMNode.getOutputProductUnit().link(currentLSTMNode.getInput(), ELinkType.RECURRENT_LINK);
						lastLSTMNode.getOutputProductUnit().link(currentLSTMNode.getInputGate(), ELinkType.RECURRENT_LINK);
						lastLSTMNode.getOutputProductUnit().link(currentLSTMNode.getForgetGate(), ELinkType.RECURRENT_LINK);
						lastLSTMNode.getOutputProductUnit().link(currentLSTMNode.getOutputGate(), ELinkType.RECURRENT_LINK);
					}
					
					lastLSTMNode = currentLSTMNode;
				}
			}
			
			for (INode node : nodes) {
				
				// We fan out links on nodes between layers (cross links)
				// also we create links for recurrent nodes.
				node.finalizeConnections();
					

			}
		
		}
		
		
	}
	
	
	
	public void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, Float ecartType){
		
		double weight = 0D;
		
		FilterIndex index = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		INode sourceNode = this.getNode(thisNode.getNodeId());
		
		if(sourceNode != null){	
				// on connecte les neurones suivant la dérivée seconde de la gaussienne
				// réalisant ainsi le filtre de Marr ou Laplacien de Gaussienne ou chapeau mexicain
				weight = linkage.getFilterValue(index, EFilterPosition.CENTER, thisNode, (IPixelNode) sourceNode);
				if(Math.abs(weight) > ecartType){
					sourceNode.link((INode) thisNode, ELinkType.REGULAR, linkage.isWeightModifiable(), weight);
				}
		}
				
		
	}
	
	public void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, SigmaWi sigmaWI) {
		
		FilterIndex index = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		INode sourceNode = this.getNode(thisNode.getNodeId());
		
		if(sourceNode != null){	
			
				sigmaWI.sum(sourceNode.getComputedOutput() * linkage.getFilterValue(index, EFilterPosition.CENTER, thisNode, (IPixelNode) sourceNode));
			
		}
		
		sigmaWI.sum(-thisNode.getBiasWeightValue());
		
		
	}
	
	
	public int getNodeCount() {
		return nodes.size();
	}

	public int getNodeCountMinusRecurrentOnes() {
		return nodes.size() - getRecurrentNodeCount();
	}

	public int getRecurrentNodeCount() {
		int recurrentNodesCount = 0;
		List<INode> nodes = (List<INode>) this.getNodes();
		
		for (INode node : nodes) {
			if (node.getNodeType() == ENodeType.RECURRENT)
				recurrentNodesCount++;
		}
		return recurrentNodesCount;
	}

	public List<INode> getNodes() {
		return nodes;
	}
	
	public List<INode> getNodes(Integer sampling) {
		
		List<INode> nodes =  this.getNodes();
		
		if(sampling == 1)
			return nodes;
		
		return nodes.stream().filter(node -> ((PixelNode) node).getX() % sampling == 0 ).filter(node -> ((PixelNode) node).getY() % sampling == 0).collect(Collectors.toList());
	}

	public INode getNode(int index) {
		return nodes.get(index);
	}
	
	

	public void setNodes(List<INode> nodes) {
		this.nodes = nodes;
	}

	public void addNode(INode node) {
		node.setNodeId(nodes.size());
		nodes.add(node);
		node.setArea(this);
		node.initGraphics();
	}
	
	public void removeNode(INode node){
		nodes.remove(node);
	}
	
	public void removeNodes(List<INode> nodes){
		nodes.removeAll(nodes);
	}

	public String toString() {
		return " Area id : " + areaId;
	}

	public Integer getAreaId() {
		return areaId;
	}

	public void setAreaId(int areaId) {
		this.areaId = areaId;
	}

	public ILayer getLayer() {
		return layer;
	}
	
	public ILayer getNextLayer() {
		return network.getLayer(layer.getLayerId() + 1);
	}
	
	public ILayer getPreviousLayer() {
		if(getLayer().isFirstLayer())
			return null;
		return network.getLayer(layer.getLayerId() - 1);
	}
	

	public IArea getLeftSibilingArea(){
		
		return getPreviousLayer().getArea(this.areaId);
	}
	
	public IArea getRightSibilingArea(){
		
		return getNextLayer().getArea(this.areaId);
	}

	public void setLayer(ILayer layer) {
		this.layer = layer;
	}
	
	

	public IArea deepCopy() {
		Area copy_area = new Area(nodes.size());
		List<INode> copy_nodes = new ArrayList<INode>(nodes);
		Collections.copy(copy_nodes, nodes);
		
		copy_area.setNodes(copy_nodes);
		copy_area.setAreaId(areaId);
		copy_area.setLayer(layer);
		
		int idx = 0;
		for(INode node : nodes){
			node.setArea(copy_area);
			copy_nodes.set(idx++, node.deepCopy());
		}
		
		return copy_area;
	}
	
	public void initGraphics() {
		if(Graphics3D.graphics3DActive)
			Graphics3D.createArea(this);
	}

	
	public String getComment() {
		return comment;
	}

	public void setComment(String comment) {
		this.comment = comment;
	}

	public EActivation getActivation() {
		return activation;
	}

	public void setActivation(EActivation activation) {
		this.activation = activation;
	}

	public ILinkage getLinkage() {
		return linkage;
	}
	
	public IFilterLinkage getFilterLinkage() {
		return (IFilterLinkage) linkage;
	}
	

	public void setLinkage(ILinkage linkage) {
		this.linkage = linkage;
	}

	public Identification getIdentification() {
		
		Integer layerId = null;
		if(this.getLayer() != null)
			layerId = this.getLayer().getLayerId();
		
		return new Identification(layerId, areaId);
	}

	@Override
	public void prePropagation() {
		getLinkage().prePropagation();
		
	}

	@Override
	public void postPropagation() {
		
		getLinkage().postPropagation();
		

		if(activation != null) {
			try {
				performer.perform();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		

	}
	
	public Boolean isShowImage() {
		return showImage;
	}

	public IActivation getPerformer() {
		return performer;
	}

	public void setPerformer(IActivation performer) {
		this.performer = performer;
	}


}
