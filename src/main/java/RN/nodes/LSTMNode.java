package RN.nodes;

import RN.ILayer;
import RN.ITester;
import RN.algoactivations.EActivation;
import RN.links.ELinkType;

/**
 * @author Eric Marchand
 *
 */
public class LSTMNode extends Node {

	public ProductNode getInputProductUnit() {
		return inputProductUnit;
	}

	public ProductNode getForgetProductUnit() {
		return forgetProductUnit;
	}

	public Node[] getMemories() {
		return memories;
	}

	private Node input = null;

	private Node inputGate = null;

	private Node forgetGate = null;

	private Node outputGate = null;

	private ProductNode inputProductUnit = null;

	private ProductNode forgetProductUnit = null;

	private Node[] memories = null;

	private ProductNode outputProductUnit = null;

	private boolean isBidirectional = false;

	public LSTMNode(int memoryCellCount) {
		super();
		nodeType = ENodeType.LSTM;

		// SYGMOID ou TANH pour input ???
		input = new Node(EActivation.TANH);
		inputGate = new Node(EActivation.SYGMOID_0_1);
		inputGate.createBias();
		forgetGate = new Node(EActivation.SYGMOID_0_1);
		forgetGate.createBias();
		outputGate = new Node(EActivation.SYGMOID_0_1);
		outputGate.createBias();

		// units have no weight
		inputProductUnit = new ProductNode();
		forgetProductUnit = new ProductNode();
		
		memories = new Node[memoryCellCount];
		for(int v=0; v < memoryCellCount; v++){
			memories[v] = new Node(EActivation.IDENTITY);
		}

		// units have no weight
		outputProductUnit = new ProductNode();

		// connect everythings

		// network input are faned out on these 4 nodes (1 input + 3 gates)
		input.link(inputProductUnit, ELinkType.REGULAR);
		inputGate.link(inputProductUnit, ELinkType.REGULAR);
		forgetGate.link(forgetProductUnit, ELinkType.REGULAR);
		outputGate.link(outputProductUnit, ELinkType.REGULAR);


		// The memory unit computes a linear function of its inputs (.)
		// The output of this unit is not squashed so that it can remember the
		// same value for many time-steps without the value decaying.
		// This value is fed back in so that the block can "remember" it (as
		// long as the forget gate allows).
		// Typically, this value is also fed into the 3 gating units to help
		// them make gating decisions.
		for (Node cellMemory : memories) {
			inputProductUnit.link(cellMemory, ELinkType.REGULAR);
			
			forgetProductUnit.link(cellMemory, ELinkType.REGULAR);
			cellMemory.link(forgetProductUnit, ELinkType.RECURRENT_LINK);

			cellMemory.link(outputProductUnit, ELinkType.REGULAR);
			// Peephole connections
//			if (isBidirectional) {
				cellMemory.link(forgetGate, ELinkType.RECURRENT_LINK);
				cellMemory.link(outputGate, ELinkType.RECURRENT_LINK);
				cellMemory.link(inputGate, ELinkType.RECURRENT_LINK);
//			}
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.Node#finalizeConnections()
	 */
	@Override
	public void finalizeConnections() {

		super.finalizeConnections();

	}

	public void sublayerFanOutLinkage(ILayer sublayer) {
		for (INode sublayerNode : sublayer.getLayerNodes()) {
			// sublayerNode.link(this, ELinkType.REGULAR);
			sublayerNode.link(input, ELinkType.REGULAR);
			sublayerNode.link(inputGate, ELinkType.REGULAR);
			sublayerNode.link(forgetGate, ELinkType.REGULAR);
			sublayerNode.link(outputGate, ELinkType.REGULAR);
		}
	}

	@Override
	public double computeOutput(boolean trainingActive) throws Exception {

		double inputResult = input.computeOutput(trainingActive);
		double inputGateResult = inputGate.computeOutput(trainingActive);
		double forgetGateResult = forgetGate.computeOutput(trainingActive);
		double outputGateResult = outputGate.computeOutput(trainingActive);
   
		double inputProductUnitResult = inputProductUnit.computeOutput(trainingActive);
		double forgetProductUnitResult = forgetProductUnit.computeOutput(trainingActive);
		
		double memoryResult = 0D;
		for (INode cellMemory : memories) {
			memoryResult += cellMemory.computeOutput(trainingActive);
		}

		double outputProductUnitResult = outputProductUnit.computeOutput(trainingActive);

		System.out.println("input " + inputResult + "  inputGate " + inputGateResult + "  forgetGate " + forgetGateResult + "  ouputGate  " + outputGateResult + "  inputProductUnit " + inputProductUnitResult + "  forgetProductUnitResult " + forgetProductUnitResult + "  outputProductUnitResult " + outputProductUnitResult);
		
		return outputProductUnitResult;
	}

	@Override
	public String getString() {

		String networkState = "";

		networkState += "input result : " + input + ITester.NEWLINE;

		networkState += "input inputGate : " + inputGate + ITester.NEWLINE;

		networkState += "input forgetGate : " + forgetGate + ITester.NEWLINE;

		networkState += "input outputGate : " + outputGate + ITester.NEWLINE;

		networkState += "input inputProductUnit : " + inputProductUnit
				+ ITester.NEWLINE;

		networkState += "input forgetProductUnit : " + forgetProductUnit
				+ ITester.NEWLINE;

		for(INode memory : memories){
			networkState += "input memory : " + memory + ITester.NEWLINE;
		}

		networkState += "input outputProductUnit : " + outputProductUnit
				+ ITester.NEWLINE;

		return super.getString();
	}

	public Node getInput() {
		return input;
	}

	public Node getInputGate() {
		return inputGate;
	}

	public Node getForgetGate() {
		return forgetGate;
	}

	public Node getOutputGate() {
		return outputGate;
	}

	public boolean isBidirectional() {
		return isBidirectional;
	}

	public void setBidirectional(boolean isBidirectional) {
		this.isBidirectional = isBidirectional;
	}

	public INode getMemory(int idx) {
		return memories[idx];
	}

	public ProductNode getOutputProductUnit() {
		return outputProductUnit;
	}

}
