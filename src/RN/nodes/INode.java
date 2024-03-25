package RN.nodes;

import java.util.List;

import RN.IArea;
import RN.Identification;
import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.links.ELinkType;
import RN.links.Link;
import RN.links.Weight;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public interface INode {


	double computeOutput(boolean playAgain) throws Exception;
	
	Link link(INode node, ELinkType type, boolean modifiable, boolean filterActive, ESamples filter );
	
	Link link(INode node, ELinkType type, boolean modifiable);
	
	Link link(INode node, ELinkType type, boolean modifiable, double weight);
	
	Link link(INode node, boolean modifiable, Weight weight);
	
	Link link(INode node, ELinkType type, Weight weight);
	
	Link link(INode node, ELinkType type);
	
	Link incomingLink(ELinkType type);

	String getString();

	ENodeType getNodeType();

	void doubleLink(INode node, ELinkType type);

	void setArea(IArea copy_area);

	Node deepCopy();

	void finalizeConnections();

	void setActivationFx(EActivation nodeActFx);

	void setActivationFxPerNode(boolean b);

	void setNodeType(ENodeType nodeType);

	Double getComputedOutput();
	
	void setComputedOutput(double computedOutput);

	Integer getNodeId();

	double getIdealOutput();

	void setError(double d);

	double getError();

	double getDerivativeValue();

	List<Link> getOutputs();

	void disconnect();

	EActivation getActivationFx();

	void addInput(Link newLink);

	void setDerivatedError(double derivatedError);

	void updateWeights(double learningRate, double alphaDeltaWeight);

	double getDerivatedError();

	void setIdealOutput(double ideal);

	Identification getIdentification();

	IArea getArea();
	
	IArea getNextArea();
	
	IArea getPreviousArea();

	Link getInput(int i);
	
	List<Link> getInputs();

	Link getBiasInput();
	
	void setNodeId(int size);

	void showImage(INode node);
	
	void newLearningCycle(int cycleCount);

	void randomizeWeights(double initWeightRange, double initWeightRange2);

	void setInnerNode(INode node);
	
	void initGraphics();

	void createBias();

	void addGraphicInterface(Pane pane);

	void initParameters(int nodeCount);

	void setEntry(Double inputValue);
	
	Double getEntry();
	
	void setBiasWeight(Weight biasWeight);
	
	Double getBiasWeightValue();

	void setBiasWeightValue(Double biasWeight);
	
	int compareOutputTo(INode comparedNode);
	
	void setDropOutActive(boolean dropOutActive);
	
	Double getDerivatedErrorSum();
	
	Double getInputValue();

	void setInputValue(Double inputValue);
	
	void setBiasPreviousDeltaWeight(Double biasPreviousDeltaWeight);
	
	int hashCode();

	boolean equals(Object obj);
	
	
}