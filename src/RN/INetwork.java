package RN;

import java.util.List;
import java.util.ListIterator;
import java.util.Set;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.links.Link;
import RN.nodes.INode;

/**
 * @author ericmarchand
 *
 */
public interface INetwork {

	
	OutputData compute(ListIterator<InputData> itr) throws Exception;

	ILayer getFirstLayer();

	ILayer getLastLayer();

	void show();

	ILayer getLayer(int i);

	List<ILayer> getLayers();

	OutputData propagation(boolean playAgain) throws Exception;

	List<INode> getAllNodes();
	
	Set<Link> getAllLinks();

	void addLayer(ILayer layer);

	void finalizeConnections();

	void init(double d, double e);

	String getString();

	void setTimeSeriesOffset(Integer timeSeriesOffset);

	Integer getTimeSeriesOffset();

	void setRecurrentNodesLinked(Boolean lateralLinkRecurrentNodes);

	Boolean isRecurrentNodesLinked();

	void setAbsoluteError(double absoluteError);
	
	double getAbsoluteError();

	INetwork deepCopy(int generationCount);
	
	void disconnectAll();
	
	String getName();

	void setName(String name);
	
	void appendName(String name);

	String geneticCodec();
	
	void initBiasWeights(double value);
	
	ENetworkImplementation getImpl();
	
	INode getNode(Identification id);

	
	/**
	 * Initialisation de parametres eventuels
	 * @param trainCycleAbsolute 
	 */
	void newLearningCycle(int trainCycleAbsolute);


}
