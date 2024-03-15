package RN.nodes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import RN.ENetworkImplementation;
import RN.Graphics3D;
import RN.IArea;
import RN.ILayer;
import RN.ITester;
import RN.Identification;
import RN.Network;
import RN.NetworkElement;
import RN.TestNetwork;
import RN.algoactivations.EActivation;
import RN.algoactivations.IActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.Linkage;
import RN.links.ELinkType;
import RN.links.Link;
import RN.links.Weight;
import javafx.scene.layout.Pane;

public class Node extends NetworkElement implements INode {

	protected Integer nodeId = null;

	protected ENodeType nodeType;
	
	protected INode innerNode = null;

	protected IArea area;

	protected double computedOutput;

	private double idealOutput;

	protected double error;

	protected List<Link> inputs;

	protected List<Link> outputs;

	protected Link biasInput;

	protected IActivation performer = null;

	private double derivatedError;

	private double aggregatedValue;

	protected boolean dropOutActive;
	
	protected EActivation activationFx;
	
	protected boolean activationFxPerNode = true;
	
	// When we have no link, input value is stored here
	protected Double inputValue;
	
	// When we have no link, bias weight is stored here
	protected Weight biasWeight;
	
	protected Double biasPreviousDeltaWeight = 0D;
	
	
	public Node() {
		this.inputs = new ArrayList<Link>();
		this.outputs = new ArrayList<Link>();
		this.nodeType = ENodeType.REGULAR;
		this.biasWeight = new Weight(0D);
	}
	
	public Node(EActivation activationFx, INode innerNode){
		this();
		this.activationFx = activationFx;
		this.activationFxPerNode = true;
		this.innerNode = innerNode;
	}
	
	public Node(EActivation activationFx) {
		this();
		this.activationFx = activationFx;
		this.activationFxPerNode = true;
	}
	
	
	public Node(INode innerNode) {
		this();
		this.innerNode = innerNode;
	}
	
	public void createBias(){
		if (network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED) {
			this.biasInput = Link.getInstance(ELinkType.REGULAR, true);
			this.biasInput.setTargetNode(this);
			this.biasInput.setUnlinkedValue(1D);
			this.biasInput.setWeightModifiable(true);
			this.biasInput.initGraphics();
		}else{
			this.biasWeight = new Weight(1D);
		}
	}

	public void disconnect(){
		
		for(Link link : inputs){
			if(link.getSourceNode() != null){
				link.getSourceNode().getOutputs().remove(link);
				link.setSourceNode(null);
			}
			link.setTargetNode(null);
			link = null;
		}
		for(Link link : outputs){
			if(link.getTargetNode() != null){
				link.getTargetNode().getInputs().remove(link);
				link.setTargetNode(null);
			}
			link.setSourceNode(null);
			
			link = null;
		}
		
		inputs.clear();
		outputs.clear();
	}

	/**
	 * Link nodes in the entier network
	 */
	public void finalizeConnections() {

		if (network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED) {
			
			ILayer layer = area.getLayer();
			Network network = layer.getNetwork();
			// Pour la couche de ce noeud (a partir de la deuxieme couche : hiddens)
			// on récupére la couche précédente (sens feedforward).
			ILayer underLayer = (layer.getLayerId() > 0 ? network.getLayers().get(layer.getLayerId() - 1) : null);
	
			if (layer.isFirstLayer()) {
	
				if (this.getNodeType() == ENodeType.RECURRENT) {
					INode sourceNode = ((RecurrentNode) this).getSourceNode();
					// weight set to 1.0
					sourceNode.link(this, ELinkType.RECURRENT_LINK, false);
				} else {
					area.getLinkage().sublayerFanOutLinkage(this, underLayer);
				}
	
			} else if (underLayer != null) {
				
				area.getLinkage().sublayerFanOutLinkage(this, underLayer);
	
			}
	
			if (layer.isDropOutLayer()) {
				this.dropOutActive = true;
			}
		
		}

	}
	
	public Link incomingLink(ELinkType type) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){		
			
			link = Link.getInstance(type, false);
			this.addInput(link);
		
		}
		
		return link;
	}
	
	public Link link(INode node, ELinkType type) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){		
			
			link = Link.getInstance(type, true);
			
			this.addOutput(link);
			
			if(node != null)
				node.addInput(link);
		
		}
		
		return link;
	}
	
	public Link link(INode node, ELinkType type, Weight weight) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){		
			
			if(type == ELinkType.SHARED)
				link = Link.getInstance(weight, true);
			else
				link = Link.getInstance(type, true);
			
			this.addOutput(link);
			
			if(node != null)
				node.addInput(link);
		
		}
		
		return link;
	}
	
	public Link link(INode node, ELinkType type, boolean modifiable) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){		
			
			link = link(node, type);
			link.setWeightModifiable(modifiable);
		
		}
		
		return link;
	}
	
	public Link link(INode node, ELinkType type, boolean modifiable, double weight) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){			
			link = link( node, type);
			link.setWeightModifiable(modifiable);
			link.setWeight(weight);
		}

		return link;
	}
	
	public Link link(INode node, boolean modifiable, Weight weight) {
		
		Link link = null;
		
		if(network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){			
			link = link( node, ELinkType.SHARED, weight);
			link.setWeightModifiable(modifiable);
		}

		return link;
	}
	

	public Link link(INode node, ELinkType type, boolean modifiable, boolean filterActive, ESamples filter ) {
		
		Link link = null;
		
		if (network == null || network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED) {
			link = link(node, type, modifiable);
			link.setFilterActive(filterActive);
			link.setFilter(filter);
		}
		
		return link;
	}
	
	public void doubleLink(INode node, ELinkType type) {
		this.link(node, type);
		node.link(this, type);
	}
	
	public void randomizeWeights(double min, double max){
		for(Link link : inputs){
			link.initWeight(min, max);
		}
		if(biasInput != null)
			biasInput.initWeight(min, max);
	}

	public void selfLink() {
		link(this, ELinkType.SELF_NODE_LINK);
	}

	/* (non-Javadoc)
	 * @see RN.INode#computeOutput(boolean)
	 */
	@Override
	public double computeOutput(boolean playAgain) throws Exception {

		double sigmaWI = 0;
		double activationFxResult;
		
		EActivation fx = activationFxPerNode ? activationFx : area.getLayer().getFunction();


		// somme des entrees pondérées - biais
		sigmaWI = area.getLinkage().getSigmaPotentials(this);
		

		// permet la généralisation (sampling)
		if (dropOutActive && TestNetwork.getInstance().isDropOutActive())
			sigmaWI *= Math.random() >= 0.5d ? 0.0D : 1.0D;	

		performer = EActivation.getPerformer(fx);

		activationFxResult = performActivationFunction(performer, sigmaWI);

		// on garde le calcul pour l'utiliser lors de la backpropagation (calcul
		// du delta des poids)
		setDerivativeValue(performDerivativeFunction(performer, sigmaWI));

		setComputedOutput(activationFxResult);
		
		if(innerNode != null)
			innerNode.showImage(this);

		return activationFxResult;
	}

	/**
	 * @param performer
	 * @param value
	 * @return
	 * @throws Exception
	 */
	public double performActivationFunction(final IActivation performer, double... values) throws Exception {

		return performer.perform(values);

	}

	/**
	 * @param performer
	 * @param value
	 * @return
	 * @throws Exception
	 */
	public double performDerivativeFunction(final IActivation performer, double... values) throws Exception {

		return performer.performDerivative(values);
	}
	
	public Double getDerivatedErrorSum(){
		
		Double derivatedErrorSum = 0D;
		Identification searchedId = getIdentification();
		
		Map<Identification, Link> outputs = Linkage.getLinksBySourceNode().get(searchedId);
		if(outputs != null){
			for(Entry<Identification, Link> output : outputs.entrySet()){
					derivatedErrorSum += output.getValue().getWeight() * network.getNode(output.getKey()).getDerivatedError();
			}
		}
		
		return derivatedErrorSum;
	}

	public IArea getArea() {
		return  area;
	}
	
	public IArea getNextArea() {
		return area.getLayer().getArea(area.getAreaId() + 1);
	}
	
	public IArea getPreviousArea() {
		return area.getLayer().getArea(area.getAreaId() - 1);
	}

	public void setArea(IArea area) {
		this.area = area;
	}

	public List<Link> getInputs() {
		return inputs;
	}

	public List<Link> getOutputs() {
		return outputs;
	}

	public Link getInput(int index) {
		return inputs.get(index);
	}

	public void setInputs(List<Link> inputs) {
		this.inputs = inputs;
	}

	public Link getBiasInput() {
		return biasInput;
	}

	public void setBiasInput(Link biasInput) {
		this.biasInput = biasInput;
	}

	public void addInput(Link link) {
		link.setInputLinkId(inputs.size());
		link.setTargetNode(this);
		inputs.add(link);
		link.initGraphics();
	}

	public void addOutput(Link link) {
		link.setOutputLinkId(outputs.size());
		link.setSourceNode(this);
		outputs.add(link);
		link.initGraphics();
	}

	public IActivation getPerformer() {
		return performer;
	}

	public void setPerformer(IActivation performer) {
		this.performer = performer;
	}

	/* (non-Javadoc)
	 * @see RN.INode#getString()
	 */
	@Override
	public String getString() {
		int jump = 0;
		String result = "            NODE : " + nodeId + " type : " + this.nodeType + ITester.NEWLINE + "                  INPUTS : "	+ (inputs.isEmpty() ? " _" : "");
		
		if(inputs.size() > 2000){
			jump = 1000;
			result += ITester.NEWLINE + ITester.NEWLINE + "        ----------------> Too much input links to print ("+ inputs.size() + "), we will print the first and last "+jump+"th links..." + ITester.NEWLINE + ITester.NEWLINE;
		}
		
		Link link = null;
		for (int id=0; id< inputs.size(); id++) {
			
			if(jump > 0 && id > jump && id < inputs.size() - jump)
				continue;
			else if(jump > 0 && id == jump)
				result += ITester.NEWLINE + "\n\n\n---------------- Jumping to the last "+jump+"th input links... ----------------\n\n\n";
			
			link = inputs.get(id);
			result += ITester.NEWLINE + "                            " + link.getString();
		}
		
		if(network.getImpl() == null || network.getImpl() == ENetworkImplementation.LINKED){
			result += ITester.NEWLINE + "                    BIAS : " + ITester.NEWLINE + "                            "
									+ (biasInput != null ? biasInput.getString() : "aucun");
		}else{
			result += ITester.NEWLINE + "                    BIAS : " + ITester.NEWLINE + "                            "
					+ (biasWeight != null && biasWeight.getWeight() != 0D ? biasWeight : "aucun");
		}
		// result += "\n                  ACTIVATION FX : " + this.function +
		// "\n                  OUTPUT :\n                      " + this.output;
		result += ITester.NEWLINE + "                   ERROR : " + ITester.NEWLINE + "                            " + this.error;
		result += ITester.NEWLINE + "                  OUTPUT : ";
		
		jump = 0;
		if(outputs.size() > 2000){
			jump = 1000;
			result += ITester.NEWLINE + ITester.NEWLINE + "        ----------------> Too much output links to print ("+ outputs.size() + "), we will print the first and last "+jump+"th links..." + ITester.NEWLINE + ITester.NEWLINE;
		}
		
		for (int id=0; id< outputs.size(); id++) {
			
			if(jump > 0 && id > jump && id < outputs.size() - jump)
				continue;
			else if(jump > 0 && id == jump)
				result += ITester.NEWLINE + "\n\n\n---------------- Jumping to the last "+jump+"th output links... ----------------\n\n\n";
			
			link = outputs.get(id);
			result += ITester.NEWLINE + "                            " + link.getString();
		}
		
		result += ITester.NEWLINE;
		return result;
	}

	public Identification getIdentification() {
		Integer layerId = null;
		Integer areaId = null;
		if(area != null){
			areaId = area.getAreaId();
			if(area.getLayer() != null)
				layerId = area.getLayer().getLayerId();
		}
		return new Identification(layerId, areaId, this.getNodeId());
	}

	public String toString() {
		return getIdentification().getIdentification();
	}

	public Integer getNodeId() {
		return nodeId;
	}

	public void setNodeId(int nodeId) {
		this.nodeId = nodeId;
	}


	public void setDerivatedError(double derivatedError) {
		this.derivatedError = derivatedError;
	}

	public double getError() {
		return error;
	}

	public double getIdealOutput() {
		return idealOutput;
	}

	public void setIdealOutput(double idealOutput) {
		this.idealOutput = idealOutput;
	}

	public double getDerivativeValue() {
		return aggregatedValue;
	}

	public void setDerivativeValue(double aggregatedValue) {
		this.aggregatedValue = aggregatedValue;
	}

	public ENodeType getNodeType() {
		return nodeType;
	}

	public void setNodeType(ENodeType nodeType) {
		this.nodeType = nodeType;
	}

	public void setError(double error) {
		this.error = error;
	}

	public void updateWeights(double learningRate, double alphaDeltaWeight) {

		Double deltaWeight = null;
		
		if(network.getImpl() == ENetworkImplementation.LINKED){
			
			for (Link input : getInputs()) {
				
				if (input.isWeightModifiable() && (input.getType() == ELinkType.REGULAR || input.getType() == ELinkType.SHARED) ) {
						deltaWeight = getDeltaWeight(false, input.getValue(), input.getPreviousDeltaWeight(), learningRate, alphaDeltaWeight);
						input.setPreviousDeltaWeight(deltaWeight);
						input.setWeight(input.getWeight() + deltaWeight);
				}
				
			}
	
			Link bias = getBiasInput();
			deltaWeight = null;
			
			if (bias != null) {
				if (bias.isWeightModifiable()) {
					deltaWeight = getDeltaWeight(true, bias.getValue(), bias.getPreviousDeltaWeight(), learningRate, alphaDeltaWeight);
					bias.setPreviousDeltaWeight(deltaWeight);
					bias.setWeight(bias.getWeight() + deltaWeight);
				}
			}
			
		}else{
			
			if(getArea().getLinkage().isWeightModifiable()){
				
				for(Entry<Identification, Link > entry : Linkage.getInputLinks(this.getIdentification()).entrySet()){
					deltaWeight = getDeltaWeight(false, entry.getValue().getValue(), entry.getValue().getPreviousDeltaWeight(), learningRate, alphaDeltaWeight);
					entry.getValue().setPreviousDeltaWeight(deltaWeight);
					entry.getValue().setWeight(entry.getValue().getWeight() + deltaWeight);
				}
				
				if(biasWeight != null && biasWeight.getWeight() != 0D){
					deltaWeight = getDeltaWeight(true, biasWeight.getWeight(), biasPreviousDeltaWeight, learningRate, alphaDeltaWeight);
					biasPreviousDeltaWeight = deltaWeight;
					biasWeight.add(deltaWeight);
				}
				
			}
			
			
		}

	}
	
	private Double getDeltaWeight(boolean bias, double value, double previousDeltaWeight, double learningRate, double alphaDeltaWeight){
		
			try {
				
				return (learningRate * derivatedError * (bias ? -1 : 1) * value) + (alphaDeltaWeight * previousDeltaWeight);
				
		} catch (Exception e) {
			System.out.println("Back-propagation erreur sur le neurone : " + this);
			throw e;
		}
	}

	public double getDerivatedError() {
		return derivatedError;
	}

	public boolean isDropOutActive() {
		return dropOutActive;
	}

	public void setDropOutActive(boolean dropOutActive) {
		this.dropOutActive = dropOutActive;
	}

	public Node deepCopy() {

		Node copy_node = new Node();
		
		 List<Link> copy_inputs  = new ArrayList<Link>(inputs);
		 List<Link> copy_outputs = new ArrayList<Link>(outputs);
		 Collections.copy(copy_inputs, inputs);
		 Collections.copy(copy_outputs, outputs);
		 
		 copy_node.setComputedOutput(computedOutput);
		 copy_node.setDerivatedError(derivatedError);
		 copy_node.setDropOutActive(dropOutActive);
		 copy_node.setError(error);
		 copy_node.setIdealOutput(idealOutput);
		 copy_node.setInputs(copy_inputs);
		 copy_node.setNodeId(nodeId);
		 copy_node.setNodeType(ENodeType.valueOf(nodeType.name()));
		 copy_node.setPerformer(performer);
		 copy_node.setOutputs(copy_outputs);
		 copy_node.setAggregatedValue(aggregatedValue);
		 if(biasInput != null){
			 copy_node.setBiasInput(biasInput.deepCopy());
		 }
		 copy_node.setArea(area);
		 
		 int idx = 0;
		for (Link link : inputs) {
			Link copy_link = link.deepCopy();
			copy_link.setTargetNode(copy_node);
			copy_inputs.set(idx++, copy_link);
			if (link.getSourceNode() != null) {
				link.getSourceNode().getOutputs().add(copy_link);
			}
		}
		 
		 idx = 0;
		for (Link link : outputs) {
			Link copy_link = link.deepCopy();
			copy_link.setSourceNode(copy_node);
			copy_outputs.set(idx++, copy_link);
			if (link.getTargetNode() != null) {
				link.getTargetNode().getInputs().add(copy_link);
			}
		}
		
		if(innerNode != null)
			copy_node.setInnerNode(innerNode.deepCopy());
		 
		 
		 return copy_node;

	}

	public Double getComputedOutput() {
		return computedOutput;
	}

	public void setComputedOutput(double computedOutput) {
		this.computedOutput = computedOutput;
		//Graphics3D.setOutputValueOnNode(this);
	}

	public double getAggregatedValue() {
		return aggregatedValue;
	}

	public void setAggregatedValue(double aggregatedValue) {
		this.aggregatedValue = aggregatedValue;
	}

	public void setOutputs(List<Link> outputs) {
		this.outputs = outputs;
	}

	public EActivation getActivationFx() {
		return activationFx;
	}

	public void setActivationFx(EActivation activationFx) {
		this.activationFx = activationFx;
	}

	public boolean isActivationFxPerNode() {
		return activationFxPerNode;
	}

	public void setActivationFxPerNode(boolean activationFxPerNode) {
		this.activationFxPerNode = activationFxPerNode;
	}

	@Override
	public void showImage(INode node) {
	}


	@Override
	public void newLearningCycle(int cycleCount) {
		// nothing to do
	}

	public INode getInnerNode() {
		return innerNode;
	}

	public void setInnerNode(INode innerNode) {
		this.innerNode = innerNode;
	}
	
	public void initGraphics() {
		if(Graphics3D.graphics3DActive)
			Graphics3D.createNode(this);
	}

	@Override
	public void addGraphicInterface(Pane pane) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParameters(int nodeCount) {
		
	}


	@Override
	public void setEntry(Double inputValue) {
		
		if(network == null || network.getImpl() == ENetworkImplementation.LINKED){
			this.getInputs().get(0).setUnlinkedValue(inputValue);
		}else{
			this.setInputValue(inputValue);
		}
		
	}
	
	@Override
	public Double getEntry() {
		
		if(network == null || network.getImpl() == ENetworkImplementation.LINKED){
			return this.getInputs().get(0).getUnlinkedValue();
		}else{
			return this.getInputValue();
		}
		
	}


	public Double getBiasWeightValue() {
		return biasWeight.getWeight();
	}

	public void setBiasWeightValue(Double biasWeight) {
		this.biasWeight.setWeight(biasWeight);
	}
	
	public void setBiasWeight(Weight biasWeight) {
		this.biasWeight = biasWeight;
	}

	@Override
	public int compareOutputTo(INode comparedNode) {
		
		if(comparedNode == null || this.getComputedOutput() > comparedNode.getComputedOutput())
			return 1;
		else if(this.getComputedOutput() < comparedNode.getComputedOutput())
			return -1;
		
		return 0;
	}

	public Double getBiasPreviousDeltaWeight() {
		return biasPreviousDeltaWeight;
	}

	public void setBiasPreviousDeltaWeight(Double biasPreviousDeltaWeight) {
		this.biasPreviousDeltaWeight = biasPreviousDeltaWeight;
	}

	public Double getInputValue() {
		return inputValue;
	}

	public void setInputValue(Double inputValue) {
		this.inputValue = inputValue;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((nodeId == null) ? 0 : nodeId.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Node other = (Node) obj;
		if (nodeId == null) {
			if (other.nodeId != null)
				return false;
		} else if (!nodeId.equals(other.nodeId))
			return false;
		return true;
	}



}
