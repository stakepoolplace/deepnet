package RN;

import java.util.List;
import java.util.Map;

import RN.algoactivations.EActivation;
import RN.links.Link;
import RN.links.Weight;
import RN.nodes.ENodeType;
import RN.nodes.INode;

/**
 * @author Eric Marchand
 * 
 */
public interface ILayer {


	boolean isLayerReccurent();

	int getAreaCount();

	int getNodeCount();

	int getNodeCountMinusRecurrentOnes();

	Integer getLayerId();

	void setLayerId(int layerId);

	void addArea(IArea area);

	void addAreas(IArea... areas);

	List<IArea> getAreas();

	IArea getArea(int index);

	void setAreas(List<IArea> areas);

	List<INode> getLayerNodes();

	List<INode> getLayerNodes(ENodeType... nodeType);

	String toString();

	Network getNetwork();

	void setNetwork(Network network);

	boolean isFirstLayer();

	boolean isLastLayer();

	Double[] propagate(boolean playAgain) throws Exception;

	void finalizeConnections();

	void setReccurent(boolean b);

	void setDropOut(Boolean dropOut);

	Boolean isDropOutLayer();

	double getLayerError();

	void setLayerError(double layerError);

	ILayer deepCopy();

	void initGraphics();

	
}