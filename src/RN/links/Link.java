package RN.links;

import java.util.Random;

import RN.Graphics3D;
import RN.NetworkContext;
import RN.NetworkElement;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 *
 */
public class Link extends NetworkElement {

	public static final long SIMPLE_DELAY = 1;
	public static final long REGULAR_DELAY = 0;
	private static Random random = new Random();

	private int inputLinkId = 0;
	private int outputLinkId = 0;
	private Weight weight = new Weight(1.0D);
	private double unlinkedValue;
	private INode targetNode;
	private INode sourceNode;
	private double previousDeltaWeight;
	private boolean filterActive = false;
	private ESamples filter;
	private boolean modifiable = true;
	private ELinkType type = ELinkType.REGULAR;
	private long fireTimeT = -1;
	
	public static Link getInstance(ELinkType type, boolean modifiable) {
		return new Link(type, null, modifiable);
	}
	
	public static Link getInstance(Weight weight, boolean modifiable) {
		return new Link(ELinkType.SHARED, weight, modifiable);
	}

	private Link() {
		this.setWeight(1.0D);
		initFireTimes();
	}

	private Link(double weight) {
		this.setWeight(weight);
		initFireTimes();
	}

	private Link(double weight, boolean isBias) {
		this.setWeight(weight);
		if (isBias)
			this.inputLinkId = -1;
		initFireTimes();
	}

	private Link(ELinkType type, Weight weight, boolean modifiable) {
		
		if(type == ELinkType.SHARED){
			this.weight = weight;
		}else{
			this.setWeight(1.0D);
		}
		this.setType(type);
		this.setWeightModifiable(modifiable);
		initFireTimes();
	}

	private Link(ELinkType type, boolean modifiable, boolean filterActive, ESamples filter) {
		this(type, null, modifiable);
		this.setFilterActive(filterActive);
		this.setFilter(filter);
	}

	public void initWeight(double min, double max) {
		this.setWeight(random.nextDouble() * (max - min) + min);
		this.setPreviousDeltaWeight(0.0);
	}

	public void initWeight(double value) {
		this.setWeight(value);
		this.setPreviousDeltaWeight(0.0);
	}

	public double getValue() {

		if (filterActive) {
			
			if (sourceNode != null) {
				return InputSample.getInstance().compute(filter, this.sourceNode.getComputedOutput());
			} else
				return InputSample.getInstance().compute(filter, this.unlinkedValue);
			
		} else {
			
			if (sourceNode != null)
				return this.sourceNode.getComputedOutput();
			else
				return this.unlinkedValue;
			
		}
	}

	public double getWeight(Double... params) {

		if (filterActive) {
			return InputSample.getInstance().compute(filter, params);
		} else {
			return weight.getWeight();
		}

	}

	public double getWeight() {
		return weight.getWeight();
	}

	public void setWeight(double weight) {
		if(this.weight == null) {
			this.weight = new Weight(weight);
		} else {
			this.weight.setWeight(weight);
		}
		Graphics3D.setWeightOnLink(this);
	}

	public String toString() {
		return (inputLinkId != -1 ? "Link" + inputLinkId : "Bias ") + " w: " + weight;
	}

	public String getString() {
		String sourceS = "";
		String targetS = "";
		if (sourceNode != null) {
			sourceS = "source:" + sourceNode.getIdentification();
		}
		if (targetNode != null) {
			targetS = " target:" + targetNode.getIdentification();
		}
		return "Fire at: " + fireTimeT + "  Link " + sourceS + targetS + "  [v : " + getValue() + "  w : " + this.weight
				+ "  previousDw : " + previousDeltaWeight + "   modifiable :" + this.modifiable + "]";
	}

	public void setInputLinkId(int linkId) {
		this.inputLinkId = linkId;
	}

	public double getPreviousDeltaWeight() {
		return previousDeltaWeight;
	}

	public void setPreviousDeltaWeight(double previousDeltaWeight) {
		this.previousDeltaWeight = previousDeltaWeight;
	}

	public void setTargetNode(INode iNode) {
		this.targetNode = iNode;

	}

	public INode getTargetNode() {
		return targetNode;
	}

	public boolean isFilterActive() {
		return filterActive;
	}

	public void setFilterActive(boolean filterActive) {
		this.filterActive = filterActive;
	}

	public ESamples getFilter() {
		return filter;
	}

	public void setFilter(ESamples filter) {
		this.filter = filter;
	}

	public boolean isWeightModifiable() {
		return modifiable;
	}

	public void setWeightModifiable(boolean modifiable) {
		this.modifiable = modifiable;
	}

	public INode getSourceNode() {
		return sourceNode;
	}

	public void setSourceNode(INode iNode) {
		this.sourceNode = iNode;
	}

	public int getOutputLinkId() {
		return outputLinkId;
	}

	public void setOutputLinkId(int outputLinkId) {
		this.outputLinkId = outputLinkId;
	}

	public ELinkType getType() {
		return type;
	}

	public void setType(ELinkType type) {
		this.type = type;
	}

	public Link deepCopy() {

		Link copy_link = new Link(weight.getWeight());
		copy_link.setFilter(filter);
		copy_link.setFilterActive(filterActive);
		copy_link.setInputLinkId(inputLinkId);
		copy_link.setOutputLinkId(outputLinkId);
		copy_link.setPreviousDeltaWeight(previousDeltaWeight);
		copy_link.setWeightModifiable(modifiable);
		copy_link.setType(type);
		copy_link.setUnlinkedValue(unlinkedValue);
		copy_link.setFireTimeT(fireTimeT);

		if (sourceNode != null)
			copy_link.setSourceNode(
					sourceNode.getArea().getLayer().getNetwork().getNode(sourceNode.getArea().getLayer().getLayerId(),
							sourceNode.getArea().getAreaId(), sourceNode.getNodeId()));

		if (targetNode != null)
			copy_link.setTargetNode(
					targetNode.getArea().getLayer().getNetwork().getNode(targetNode.getArea().getLayer().getLayerId(),
							targetNode.getArea().getAreaId(), targetNode.getNodeId()));

		return copy_link;
	}

	public double getUnlinkedValue() {
		return unlinkedValue;
	}

	public void setUnlinkedValue(double unlinkedValue) {
		this.unlinkedValue = unlinkedValue;
	}

	public void synchFutureFire() {
		fireTimeT = fireTimeT + NetworkContext.INCREMENTUM + getActionPotentialDelay(type);
	}

	private long getActionPotentialDelay(ELinkType linkType) {

		// if(linkType == ELinkType.REGULAR)
		// return REGULAR_DELAY;
		// else if(linkType == ELinkType.RECURRENT_LATERAL_LINK)
		// return REGULAR_DELAY;
		// else if(linkType == ELinkType.RECURRENT_LINK)
		// return REGULAR_DELAY;
		// else if(linkType == ELinkType.SELF_NODE_LINK)
		// return REGULAR_DELAY;
		// else

		if (linkType == ELinkType.LAGGED_LINK)
			return SIMPLE_DELAY;
		else
			return REGULAR_DELAY;

	}

	public long getFireTimeT() {
		return fireTimeT;
	}

	public void setFireTimeT(long fireTimeT) {
		this.fireTimeT = fireTimeT;
	}

	public void initFireTimes() {

		if (this.getType() == ELinkType.LAGGED_LINK)
			this.setFireTimeT(Link.SIMPLE_DELAY);
		else
			this.setFireTimeT(Link.REGULAR_DELAY);

	}
	
	public boolean isBias(){
		return this.inputLinkId == -1;
	}

	public void initGraphics() {
		if(Graphics3D.graphics3DActive)
			Graphics3D.createLink(this);
	}



}
