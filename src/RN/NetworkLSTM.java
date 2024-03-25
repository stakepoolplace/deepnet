package RN;
import java.util.List;
import java.util.ListIterator;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.nodes.INode;


/**
 * @author Eric Marchand
 * 
 */
public class NetworkLSTM implements INetwork{


	@Override
	public ILayer getFirstLayer() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ILayer getLastLayer() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void show() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public ILayer getLayer(int i) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<ILayer> getLayers() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public OutputData propagation(boolean trainingActive) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<INode> getAllNodes() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void addLayer(ILayer layer) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void finalizeConnections() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void init(double d, double e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getString() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public OutputData compute(ListIterator<InputData> itr) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setTimeSeriesOffset(Integer timeSeriesOffset) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getTimeSeriesOffset() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setRecurrentNodesLinked(Boolean lateralLinkRecurrentNodes) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Boolean isRecurrentNodesLinked() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setAbsoluteError(double absoluteError) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getAbsoluteError() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public void disconnectAll() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void appendName(String name) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public INetwork deepCopy(int generationCount) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String geneticCodec() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void newLearningCycle(int cycleCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initBiasWeights(double value) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public ENetworkImplementation getImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public INode getNode(Identification id) {
		// TODO Auto-generated method stub
		return null;
	}

}
