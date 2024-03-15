package RN.nodes;

import java.util.List;

import RN.IAreaSquare;
import RN.Identification;
import RN.dataset.Coordinate;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.vision.Gradient;
import RN.links.ELinkType;
import RN.links.Link;
import RN.links.Weight;
import javafx.scene.layout.Pane;

public interface IPixelNode {
	
	Coordinate getCoordinate();
	
	Integer getNodeId();
	
	IAreaSquare getAreaSquare();
	
	void addGraphicInterface(Pane pane);
	
	List<Link> getInputs();
	
	Double getComputedOutput();
	
	void setComputedOutput(double computedOutput);
	
	void setEntry(Double inputValue);
	
	Double getInputValue();
	
	Link getInput(int i);
	
	Link link(INode node, ELinkType type, boolean modifiable, boolean filterActive, ESamples filter );
	
	Link link(INode node, ELinkType type, boolean modifiable);
	
	Link link(INode node, ELinkType type, boolean modifiable, double weight);
	
	Link link(INode node, boolean modifiable, Weight weight);
	
	Link link(INode node, ELinkType type, Weight weight);
	
	Link link(INode node, ELinkType type);
	
	Double getBiasWeightValue();
	
	Double compareOutputTo(IPixelNode comparedNode);
	
	int compareOutputTo(Double value);
	
	Identification getIdentification();
	
	IAreaSquare getNextAreaSquare();
	
	IAreaSquare getPreviousAreaSquare();
	
	int getX();

	int getY();
	
	double getR();
	
	double getP(double base);
	
	double getTheta();
	
	void setTheta(Double theta);

	void setP(Double p);
	
	void setR(Double r);
	
	IPixelNode getLeft();
	
	IPixelNode getRight();
	
	IPixelNode getUp();
	
	IPixelNode getUpLeft();
	
	IPixelNode getUpRight();
	
	IPixelNode getDown();
	
	IPixelNode getDownLeft();
	
	IPixelNode getDownRight();
	
	Gradient getGradient();
	
	Double distance(IPixelNode n1, IPixelNode n2);
	
	Double distance(IPixelNode n);

}
