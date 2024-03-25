package RN;

import java.util.ArrayList;
import java.util.List;

import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.ELinkage;
import RN.linkage.ELinkageBetweenAreas;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.utils.ReflexionUtils;

/**
 * @author Eric Marchand
 * 
 */
public class NetworkContext {
	
	private static NetworkContext context = null;
	
	private long clock = -1L;
	
	public static long INCREMENTUM = 1L;
	
	private String networkType;
	
	private String[] kinds;

	private Integer[] layers;

	private Integer[] areas;
	
	private String[] areasType;
	
	private EBoolean[] areasImage;

	private Integer[] nodes;
	
	private Double[] nodeBiasWeights;

	private String[] nodeTypes;

	private EActivation[] nodeActivations;

	private Boolean[] nodeRecurrents;

	private Integer[] linkSourceTarget;

	private List<Integer[]> links;

	private ESamples[] filters;
	
	private String[] labels;
	
	private Integer[] samplings;
	
	private Integer[] scalings;

	private ELinkage[] linkages;
	
	private ELinkageBetweenAreas[] linkageBetweenAreas;
	
	private Integer[][] linkageTargetedAreas;
	
	private Double[][] linkageOptParams;

	private EBoolean[] linkageWeightModifiables;
	

	public void setClock(long clock) {
		this.clock = clock;
	}

	public NetworkContext() {
	}

	public ESamples getFilter(int columnIndex) {
		return filters[columnIndex];
	}

	public String getKind(int columnIndex) {
		return kinds[columnIndex];
	}
	
	public void addNetworkType(String value) {
		this.networkType = value;
	}

	public void addKind(String value, int idx) {
		this.kinds[idx] = value;
	}

	public void addLayer(Integer value, int idx) {
		this.layers[idx] = value;
	}

	public void addLabel(String value, int idx) {
		this.labels[idx] = value;
	}
	
	public void addSample(Integer value, int idx) {
		this.samplings[idx] = value;
	}
	
	public void addScale(Integer value, int idx) {
		this.scalings[idx] = value;
	}
	
	public void addArea(Integer value, int idx) {
		this.areas[idx] = value;
	}
	
	public void addAreaType(String value, int idx) {
		this.areasType[idx] = value;
	}
	
	public void addAreaImage(EBoolean value, int idx) {
		this.areasImage[idx] = value;
	}
	

	public void addNode(Integer value, int idx) {
		this.nodes[idx] = value;
	}

	public void addNodeType(String value, int idx) {
		this.nodeTypes[idx] = value;
	}
	
	public void addNodeBiasWeight(Double value, int idx){
		this.nodeBiasWeights[idx] = value;
	}

	public void addNodeActivation(String value, int idx) {
		this.nodeActivations[idx] = EActivation.valueOf(value);
	}
	
	public void addLinkage(String value, int idx) {
		this.linkages[idx] = ELinkage.valueOf(value);
	}
	
	public void addLinkageBetweenAreas(String value, int idx) {
		this.linkageBetweenAreas[idx] = ELinkageBetweenAreas.valueOf(value);
	}
	
	public void addLinkageTargetedArea(Integer[] value, int idx) {
		this.linkageTargetedAreas[idx] = value;
	}
	
	
	public void addLinkageOptParams(Double[] value, int idx) {
		this.linkageOptParams[idx] = value;
	}
	
	public void addLinkageWeightModifiable(EBoolean value, int idx) {
		this.linkageWeightModifiables[idx] = value;
	}

	public void addNodeRecurrent(Boolean value, int idx) {
		this.nodeRecurrents[idx] = value;
	}

	public void addLink(int sourceValue, int targetValue) {
		linkSourceTarget = new Integer[2];
		linkSourceTarget[0] = sourceValue;
		linkSourceTarget[1] = targetValue;
		this.links.add(linkSourceTarget);
	}

	public void addFilter(String value, int idx) {
		this.filters[idx] = ESamples.valueOf(value);
	}

	public void initLinks(int size) {
		links = new ArrayList<Integer[]>(size);
	}

	public void initLayers(int size) {
		layers = new Integer[size];
	}

	public void initAreas(int size) {
		areas = new Integer[size];
	}
	
	public void initAreasType(int size) {
		areasType = new String[size];
	}
	
	public void initAreasImage(int size) {
		areasImage = new EBoolean[size];
	}
	
	public void initNodes(int size) {
		nodes = new Integer[size];
	}

	public void initNodeTypes(int size) {
		nodeTypes = new String[size];
	}
	
	public void initNodeBiasWeight(int size){
		nodeBiasWeights = new Double[size];
	}

	public void initNodeActivations(int size) {
		nodeActivations = new EActivation[size];
	}

	public void initNodeRecurrents(int size) {
		nodeRecurrents = new Boolean[size];
	}

	public void initKinds(int size) {
		kinds = new String[size];
	}

	public void initFilters(int size) {
		filters = new ESamples[size];
	}

	public String[] getKinds() {
		return kinds;
	}

	public void setKinds(String[] kinds) {
		this.kinds = kinds;
	}

	public Integer[] getLayers() {
		return layers;
	}

	public void setLayers(Integer[] layers) {
		this.layers = layers;
	}

	public Integer[] getAreas() {
		return areas;
	}

	public void setAreas(Integer[] areas) {
		this.areas = areas;
	}

	public Integer[] getNodes() {
		return nodes;
	}

	public void setNodes(Integer[] nodes) {
		this.nodes = nodes;
	}

	public String[] getNodeTypes() {
		return nodeTypes;
	}

	public void setNodeTypes(String[] nodeTypes) {
		this.nodeTypes = nodeTypes;
	}

	public EActivation[] getNodeActivations() {
		return nodeActivations;
	}

	public void setNodeActivations(EActivation[] nodeActivations) {
		this.nodeActivations = nodeActivations;
	}

	public Boolean[] getNodeRecurrents() {
		return nodeRecurrents;
	}

	public void setNodeRecurrents(Boolean[] nodeRecurrents) {
		this.nodeRecurrents = nodeRecurrents;
	}

	public Integer[] getLinkSourceTarget() {
		return linkSourceTarget;
	}

	public void setLinkSourceTarget(Integer[] linkSourceTarget) {
		this.linkSourceTarget = linkSourceTarget;
	}

	public List<Integer[]> getLinks() {
		return links;
	}

	public void setLinks(List<Integer[]> links) {
		this.links = links;
	}

	public ESamples[] getFilters() {
		return filters;
	}

	public void setFilters(ESamples[] filters) {
		this.filters = filters;
	}

	public int getColumnCountByLayer(Integer layerIdCounted) {
		int count = 0;
		for (int idx = 0; idx < layers.length; idx++) {
			Integer layerId = layers[idx];
			if(layerIdCounted.equals(layerId))
				count++;
		}

		return count;
	}
	
	public int getColumnCountByLayerAndArea(Integer layerIdCounted, Integer idxArea) {
		int count = 0;
		Integer layerId = null;
		Integer areaId = null; 
		for (int idx = 0; idx < layers.length; idx++) {
			layerId = layers[idx];
			areaId = areas[idx];
			if(layerIdCounted.equals(layerId) && idxArea.equals(areaId))
				count++;
		}

		return count;
	}
	
	
	public int getNodeSumByLayerAreaAndKind(Integer layerIdCounted, Integer areaIdCounted, String kindCounted) {
		int sum = 0;
		for (int idx = 0; idx < kinds.length; idx++) {
			if(kinds[idx] == null)
				break;
			Integer layerId = layers[idx];
			Integer areaId = areas[idx];
			String kind = kinds[idx];
			int nodeCount = nodes[idx];
			if (layerIdCounted != null && areaIdCounted != null && kindCounted.equals(kind) && layerIdCounted.equals(layerId) && areaIdCounted.equals(areaId)) {
				sum += nodeCount;
			}
		}

		return sum;
	}
	
	public int getNodeSumByLayerAndKind(Integer layerIdCounted, String kindCounted) {
		int sum = 0;
		for (int idx = 0; idx < kinds.length; idx++) {
			if(kinds[idx] == null)
				break;
			int layerId = layers[idx];
			String kind = kinds[idx];
			int nodeCount = nodes[idx];
			if (layerIdCounted != null && kindCounted.equals(kind) && layerIdCounted.equals(layerId)) {
				sum += nodeCount;
			}
		}

		return sum;
	}
	
	public int getNodeSumByKind(String kindCounted) {
		int sum = 0;
		for (int idx = 0; idx < kinds.length; idx++) {
			if(kinds[idx] == null)
				break;
			String kind = kinds[idx];
			int nodeCount = nodes[idx];
			if (kindCounted.equals(kind)) {
				sum += nodeCount;
			}
		}

		return sum;
	}
	
	public int getLayerCount(){
		int max = 0;
		for(Integer layerId : layers)
			if(layerId != null && layerId > max)
				max = layerId;
		
		return max + 1;
	}
	
	public int getAreaCount(int layerId){
		
		int max = 0;
		int idx = 0;
		for(Integer areaId : areas){
			
			if(layers.length <= idx && layers[idx] == layerId && areaId != null && areaId > max)
				max = areaId;
			
			idx++;
		}
		
		return max + 1;
	}
	
	
	public INetwork newNetwork(String networkName) {
		
		INetwork network = null;
		if(ENetworkImplementation.UNLINKED.name().equalsIgnoreCase(networkType)){
			network = Network.newInstance(ENetworkImplementation.UNLINKED);
		}else{
			network = Network.newInstance(ENetworkImplementation.LINKED);
		}
		
		network.setName(networkName);

		// Neuralware (2001) approach
//		optimizedNumHiddens = getTrainingVectorNumber() / (5 * (getInputSize() + getOutputSize()));
//		optimizedNumHiddens = Math.max(5, optimizedNumHiddens);
		Layer layer = null;
		IArea area = null;
		int nodeNb = 0;
		int colOffset = 0;
		EAreaType areaType = null;
		EBoolean areaImage = null;
		Class[] areaParamClasses = null;
		Object[] areaParamValues = null;
		int idxCol = 0;
		List<INode> nodeList = null;

		for(int idx = 0; idx < getLayerCount(); idx++){
			
			layer = new Layer(EActivation.IDENTITY);
			network.addLayer(layer);
			
			for(int idxArea = 0; idxArea < getAreaCount(idx); idxArea++){
				
				idxCol = colOffset + idxArea;
				
				nodeNb = getNodeSumByLayerAreaAndKind(idx, idxArea, kinds[idxCol]);
				
				areaType = EAreaType.valueOf(areasType[idxCol].trim());
				areaImage = areasImage[idxCol];
				areaParamClasses = new Class[1];
				areaParamValues = new Object[1];
				if(areaImage != null){
					areaParamClasses = new Class[2];
					areaParamValues = new Object[2];
					areaParamClasses[1] = boolean.class;
					areaParamValues[1] = areaImage.getValue();
				}
				areaParamClasses[0] = int.class;
				areaParamValues[0] = nodeNb;
				area = ReflexionUtils.newClass(areaType.getClassPath(), areaParamClasses, areaParamValues);
				
				layer.addArea(area);
				
				area.configureLinkage(linkages[idxCol], linkageBetweenAreas[idxCol], linkageTargetedAreas[idxCol], null, (samplings[idxCol] == null ? 1 : samplings[idxCol]), linkageWeightModifiables[idxCol].getValue(), linkageOptParams[idxCol]);
				
				for(int idxByArea = 0; idxByArea < getColumnCountByLayerAndArea(idx, idxArea); idxByArea++){
					
					area.configureNode(nodeBiasWeights[idxCol] != null, nodeActivations[idxCol], ENodeType.valueOf(nodeTypes[idxCol]));
					
					nodeList = area.createNodes(nodes[idxCol]);
					
					if(areaType == EAreaType.SQUARE && scalings[idxCol] != null){
						((IAreaSquare) area).getImageArea().scaleImage(scalings[idxCol]);
					}
					
					ENodeType nodeType = ENodeType.valueOf(nodeTypes[idxCol].trim());
					Boolean nodeRecurrent = nodeRecurrents[idxCol] == null ? Boolean.FALSE : Boolean.TRUE;
					layer.setDropOut(false);
					// TODO a changer par node.setReccurent()
					layer.setReccurent(nodeRecurrent);
					
					for(INode node : nodeList){
						
						node.setActivationFxPerNode(true);
						node.setNodeType(nodeType);
						
						if(nodeBiasWeights[idxCol] != null){
							if(ENetworkImplementation.UNLINKED.name().equalsIgnoreCase(networkType))
								node.setBiasWeightValue(nodeBiasWeights[idxCol]);
							else
								node.getBiasInput().setWeight(nodeBiasWeights[idxCol]);
						}
						

					}
					idxCol++;
				}
				
			}
			
			colOffset += getColumnCountByLayer(idx);
			
			
		}
		
		network.setTimeSeriesOffset(1);
		network.setRecurrentNodesLinked(false);
		network.setName(network.getName() + network.geneticCodec());
		
		return network;
	}

	public void initData(int size) {
		DataSeries.getInstance().clearSeries();
	}
	
	public long incrementClock(){
		return clock = clock + INCREMENTUM;
	}

	public long getClock() {
		return clock;
	}

	public String[] getLabels() {
		return labels;
	}
	
	public String getLabel(int idx){
		return labels[idx];
	}

	public void setLabels(String[] labels) {
		this.labels = labels;
	}

	public void initLabels(int size) {
		this.labels = new String[size];
	}
	
	public void initSampling(int size) {
		this.samplings = new Integer[size];
	}
	
	public void initScale(int size) {
		this.scalings = new Integer[size];
	}
	
	public void initLinkages(int size) {
		this.linkages = new ELinkage[size];
	}
	
	public void initLinkageBetweenAreas(int size) {
		this.linkageBetweenAreas = new ELinkageBetweenAreas[size];
	}
	
	public void initLinkageTargetedArea(int size) {
		this.linkageTargetedAreas = new Integer[size][];
	}
	
	
	public void initLinkageOptParams(int size) {
		this.linkageOptParams = new Double[size][];
	}
	
	
	public void initLinkageWeightModifiables(int size) {
		this.linkageWeightModifiables = new EBoolean[size];
	}
	
	public static NetworkContext getContext() {
		
		if(context == null){
			context = new NetworkContext();
		}
		
		return context;
	}

	public static void setContext(NetworkContext context) {
		NetworkContext.context = context;
	}

	public Integer[] getSamplings() {
		return samplings;
	}
	
	public Integer getSamplings(int idx) {
		
		if(samplings == null || samplings[idx] == null)
			return 1;
		
		return samplings[idx];
	}

	public void setSamplings(Integer[] samplings) {
		this.samplings = samplings;
	}

	public Double[] getNodeBiasWeights() {
		return nodeBiasWeights;
	}

	public void setNodeBiasWeights(Double[] nodeBiasWeights) {
		this.nodeBiasWeights = nodeBiasWeights;
	}

	public String getNetworkType() {
		return networkType;
	}

	public void setNetworkType(String networkType) {
		this.networkType = networkType;
	}

	public ELinkageBetweenAreas[] getNodeLinkageBetweenAreas() {
		return linkageBetweenAreas;
	}

	public void setNodeLinkageBetweenAreas(ELinkageBetweenAreas[] nodeLinkageBetweenAreas) {
		this.linkageBetweenAreas = nodeLinkageBetweenAreas;
	}

	public Integer[][] getNodeLinkageTargetedAreas() {
		return linkageTargetedAreas;
	}

	public void setNodeLinkageTargetedAreas(Integer[][] nodeLinkageTargetedAreas) {
		this.linkageTargetedAreas = nodeLinkageTargetedAreas;
	}


	
}
