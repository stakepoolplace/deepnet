package RN;

import java.util.List;
import java.util.ListIterator;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.nodes.INode;
import dmonner.xlbp.Component;
import dmonner.xlbp.Network;
import dmonner.xlbp.NetworkCopier;
import dmonner.xlbp.NetworkStringBuilder;
import dmonner.xlbp.UpstreamComponent;
import dmonner.xlbp.WeightInitializer;
import dmonner.xlbp.WeightUpdaterType;
import dmonner.xlbp.layer.InputLayer;
import dmonner.xlbp.layer.TargetLayer;

/**
 * @author Eric Marchand
 * 
 */
public class NetLSTMNetwork extends Network implements INetwork {

	public NetLSTMNetwork(String name) {
		super(name);
	}

	@Override
	public void activateTest() {
		// TODO Auto-generated method stub
		super.activateTest();
	}

	@Override
	public void activateTrain() {
		// TODO Auto-generated method stub
		super.activateTrain();
	}

	@Override
	public void add(Component component) {
		// TODO Auto-generated method stub
		super.add(component);
	}

	@Override
	public void add(Component component, boolean activate, boolean train, boolean entry, boolean exit) {
		// TODO Auto-generated method stub
		super.add(component, activate, train, entry, exit);
	}

	@Override
	public void addActivateOnly(Component component) {
		// TODO Auto-generated method stub
		super.addActivateOnly(component);
	}

	@Override
	public void addTrainOnly(Component component) {
		// TODO Auto-generated method stub
		super.addTrainOnly(component);
	}

	@Override
	public void addUpstream(UpstreamComponent upstream) {
		// TODO Auto-generated method stub
		super.addUpstream(upstream);
	}

	@Override
	public void addUpstream(UpstreamComponent upstream, boolean weighted) {
		// TODO Auto-generated method stub
		super.addUpstream(upstream, weighted);
	}

	@Override
	public void addUpstreamWeights(UpstreamComponent upstream) {
		// TODO Auto-generated method stub
		super.addUpstreamWeights(upstream);
	}

	@Override
	public void build() {
		// TODO Auto-generated method stub
		super.build();
	}

	@Override
	public void clear() {
		// TODO Auto-generated method stub
		super.clear();
	}

	@Override
	public void clearActivations() {
		// TODO Auto-generated method stub
		super.clearActivations();
	}

	@Override
	public void clearEligibilities() {
		// TODO Auto-generated method stub
		super.clearEligibilities();
	}

	@Override
	public void clearInputs() {
		// TODO Auto-generated method stub
		super.clearInputs();
	}

	@Override
	public void clearResponsibilities() {
		// TODO Auto-generated method stub
		super.clearResponsibilities();
	}

	@Override
	public int compareTo(Component that) {
		// TODO Auto-generated method stub
		return super.compareTo(that);
	}

	@Override
	public Network copy(NetworkCopier copier) {
		// TODO Auto-generated method stub
		return super.copy(copier);
	}

	@Override
	public Network copy(String suffix) {
		// TODO Auto-generated method stub
		return super.copy(suffix);
	}

	@Override
	public Network copy(String suffix, boolean copyState, boolean copyWeights) {
		// TODO Auto-generated method stub
		return super.copy(suffix, copyState, copyWeights);
	}

	@Override
	public Network copy(String prefix, String suffix, boolean copyState, boolean copyWeights) {
		// TODO Auto-generated method stub
		return super.copy(prefix, suffix, copyState, copyWeights);
	}

	@Override
	public void copyConnectivityFrom(Component comp, NetworkCopier copier) {
		// TODO Auto-generated method stub
		super.copyConnectivityFrom(comp, copier);
	}

	@Override
	public void ensureActivateCapacity(int cActivate) {
		// TODO Auto-generated method stub
		super.ensureActivateCapacity(cActivate);
	}

	@Override
	public void ensureAllCapacity(int cAll) {
		// TODO Auto-generated method stub
		super.ensureAllCapacity(cAll);
	}

	@Override
	public void ensureDirectEntryCapacity(int cEntry) {
		// TODO Auto-generated method stub
		super.ensureDirectEntryCapacity(cEntry);
	}

	@Override
	public void ensureDirectExitCapacity(int cExit) {
		// TODO Auto-generated method stub
		super.ensureDirectExitCapacity(cExit);
	}

	@Override
	public void ensureInputCapacity(int cInputs) {
		// TODO Auto-generated method stub
		super.ensureInputCapacity(cInputs);
	}

	@Override
	public void ensureSubnetCapacity(int cSubnets) {
		// TODO Auto-generated method stub
		super.ensureSubnetCapacity(cSubnets);
	}

	@Override
	public void ensureTargetCapacity(int cTargets) {
		// TODO Auto-generated method stub
		super.ensureTargetCapacity(cTargets);
	}

	@Override
	public void ensureTrainCapacity(int cTrain) {
		// TODO Auto-generated method stub
		super.ensureTrainCapacity(cTrain);
	}

	@Override
	public void ensureWeightedEntryCapacity(int cEntry) {
		// TODO Auto-generated method stub
		super.ensureWeightedEntryCapacity(cEntry);
	}

	@Override
	public Component getActivate(int index) {
		// TODO Auto-generated method stub
		return super.getActivate(index);
	}

	@Override
	public int getActivateSize() {
		// TODO Auto-generated method stub
		return super.getActivateSize();
	}

	@Override
	public Component getComponent(int index) {
		// TODO Auto-generated method stub
		return super.getComponent(index);
	}

	@Override
	public Component getComponentByName(String name) {
		// TODO Auto-generated method stub
		return super.getComponentByName(name);
	}

	@Override
	public Component[] getComponents() {
		// TODO Auto-generated method stub
		return super.getComponents();
	}

	@Override
	public UpstreamComponent getExitPoint() {
		// TODO Auto-generated method stub
		return super.getExitPoint();
	}

	@Override
	public UpstreamComponent getExitPoint(int i) {
		// TODO Auto-generated method stub
		return super.getExitPoint(i);
	}

	@Override
	public InputLayer getInputLayer() {
		// TODO Auto-generated method stub
		return super.getInputLayer();
	}

	@Override
	public InputLayer getInputLayer(int index) {
		// TODO Auto-generated method stub
		return super.getInputLayer(index);
	}

	@Override
	public InputLayer[] getInputLayers() {
		// TODO Auto-generated method stub
		return super.getInputLayers();
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return super.getName();
	}

	@Override
	public int getNExitPoints() {
		// TODO Auto-generated method stub
		return super.getNExitPoints();
	}

	@Override
	public TargetLayer getTargetLayer() {
		// TODO Auto-generated method stub
		return super.getTargetLayer();
	}

	@Override
	public TargetLayer getTargetLayer(int index) {
		// TODO Auto-generated method stub
		return super.getTargetLayer(index);
	}

	@Override
	public TargetLayer[] getTargetLayers() {
		// TODO Auto-generated method stub
		return super.getTargetLayers();
	}

	@Override
	public Component getTrain(int index) {
		// TODO Auto-generated method stub
		return super.getTrain(index);
	}

	@Override
	public int getTrainSize() {
		// TODO Auto-generated method stub
		return super.getTrainSize();
	}

	@Override
	public boolean isBuilt() {
		// TODO Auto-generated method stub
		return super.isBuilt();
	}

	@Override
	public int nInput() {
		// TODO Auto-generated method stub
		return super.nInput();
	}

	@Override
	public int nTarget() {
		// TODO Auto-generated method stub
		return super.nTarget();
	}

	@Override
	public int nWeights() {
		// TODO Auto-generated method stub
		return super.nWeights();
	}

	@Override
	public int nWeightsDeep() {
		// TODO Auto-generated method stub
		return super.nWeightsDeep();
	}

	@Override
	public boolean optimize() {
		// TODO Auto-generated method stub
		return super.optimize();
	}

	@Override
	public void processBatch() {
		// TODO Auto-generated method stub
		super.processBatch();
	}

	@Override
	public void rebuild() {
		// TODO Auto-generated method stub
		super.rebuild();
	}

	@Override
	public void setActivateSize(int nActivate) {
		// TODO Auto-generated method stub
		super.setActivateSize(nActivate);
	}

	@Override
	public void setAllSize(int nAll) {
		// TODO Auto-generated method stub
		super.setAllSize(nAll);
	}

	@Override
	public void setInput(float[] input) {
		// TODO Auto-generated method stub
		super.setInput(input);
	}

	@Override
	public void setInput(int index, float[] activations) {
		// TODO Auto-generated method stub
		super.setInput(index, activations);
	}

	@Override
	public void setInputSize(int nInputs) {
		// TODO Auto-generated method stub
		super.setInputSize(nInputs);
	}

	@Override
	public void setTarget(float[] target) {
		// TODO Auto-generated method stub
		super.setTarget(target);
	}

	@Override
	public void setTarget(int index, float[] activations) {
		// TODO Auto-generated method stub
		super.setTarget(index, activations);
	}

	@Override
	public void setTargetSize(int nTargets) {
		// TODO Auto-generated method stub
		super.setTargetSize(nTargets);
	}

	@Override
	public void setTrainSize(int nTrain) {
		// TODO Auto-generated method stub
		super.setTrainSize(nTrain);
	}

	@Override
	public void setWeightInitializer(WeightInitializer win) {
		// TODO Auto-generated method stub
		super.setWeightInitializer(win);
	}

	@Override
	public void setWeightUpdaterType(WeightUpdaterType wut) {
		// TODO Auto-generated method stub
		super.setWeightUpdaterType(wut);
	}

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return super.size();
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return super.toString();
	}

	@Override
	public void toString(NetworkStringBuilder sb) {
		// TODO Auto-generated method stub
		super.toString(sb);
	}

	@Override
	public String toString(String show) {
		// TODO Auto-generated method stub
		return super.toString(show);
	}

	@Override
	public void unbuild() {
		// TODO Auto-generated method stub
		super.unbuild();
	}

	@Override
	public void updateEligibilities() {
		// TODO Auto-generated method stub
		super.updateEligibilities();
	}

	@Override
	public void updateResponsibilities() {
		// TODO Auto-generated method stub
		super.updateResponsibilities();
	}

	@Override
	public void updateWeights() {
		// TODO Auto-generated method stub
		super.updateWeights();
	}

	@Override
	public int hashCode() {
		// TODO Auto-generated method stub
		return super.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		// TODO Auto-generated method stub
		return super.equals(obj);
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return super.clone();
	}

	@Override
	protected void finalize() throws Throwable {
		// TODO Auto-generated method stub
		super.finalize();
	}

	@Override
	public OutputData compute(ListIterator<InputData> itr) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

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
	public INetwork deepCopy(int generationCount) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void disconnectAll() {
		// TODO Auto-generated method stub
		
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
	public String geneticCodec() {
		// TODO Auto-generated method stub
		return null;
	}



	@Override
	public void newLearningCycle(int trainCycleAbsolute) {
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
