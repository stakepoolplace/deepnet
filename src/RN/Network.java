package RN;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.dataset.OutputDataList;
import RN.fxml.controllers.NN;
import RN.genetic.Genetic;
import RN.linkage.Linkage;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.utils.StatUtils;

/**
 * @author Eric Marchand
 * 
 */
public class Network extends NetworkElement implements Serializable, INetwork{

	private List<ILayer> layers = null;

	private OutputDataList outputDataList = new OutputDataList();

	private Integer recurrentLayerId = null;

	private String name = null;

	private Integer timeSeriesOffset = null;

	private Boolean recurrentNodesLinked = Boolean.FALSE;
	
	private double absoluteError = 0.0D;

	private ENetworkImplementation impl = ENetworkImplementation.LINKED;
	
	
	private Network() {
		layers = new ArrayList<ILayer>();
	}
	
	private Network(ENetworkImplementation impl) {
		layers = new ArrayList<ILayer>();
		
		if(impl != null)
			this.impl  = impl;
	}
	
	public static Network getInstance(){
		return getInstance(null);
	}
	
	public static Network getInstance(ENetworkImplementation impl){
		if(network == null){
			network = new Network(impl);
		}
		return network;
	}
	
	public static Network newInstance(ENetworkImplementation impl){
		network = new Network(impl);
		return network;
	}

	

	public void addLayer(ILayer layer) {
		layer.setLayerId(layers.size());
		layers.add(layer);
		layer.setNetwork(this);
		layer.initGraphics();
	}
	
	public void addLayer(ILayer... layers) {
		for(ILayer lay: layers)
			addLayer(lay);
	}

	public ILayer getLayer(int idLayer) {
		return layers.get(idLayer);
	}

	public List<ILayer> getLayers() {
		return layers;
	}

	public void setLayers(List<ILayer> layers) {
		this.layers = layers;
	}

	/**
	 * 
	 */
	public void finalizeConnections() {

		if (network.getImpl() == ENetworkImplementation.LINKED) {

			System.out.println("Begin finalize network connections...");
			
			for (ILayer layer : layers) {

				// Create recurrent nodes in input layer if needed
				layer.finalizeConnections();
				
				// finalize connections between areas themselves
				finalizeAreasConnections(layer);
				
			}
			
			System.out.println("End finalize network connections.");

		} else {

			System.out.println(
					"Network created without any link object, method finalizeConnections() is useless in this case.");
		}


		// Every nodes are linked, let's init links for future firings
		// Set<Link> uniqLinks = new LinkedHashSet<Link>();
		// for(INode node : getAllNodes()){
		// uniqLinks.addAll(node.getInputs());
		// uniqLinks.addAll(node.getOutputs());
		// if(node.getBiasInput() != null)
		// uniqLinks.add(node.getBiasInput());
		// }
		// for(Link link : uniqLinks){
		// link.initFireTimes();
		// }

	}

	private void finalizeAreasConnections(ILayer layer) {
		List<IArea> areas = layer.getAreas();

		for (IArea area : areas) {

			// Linkage is done following ELinkage type.
			// recurrent nodes linked themselves and finalize
			// connections between LSTM nodes (timeseries).
			area.finalizeConnections();

		}
	}

	/**
	 * 
	 */
	public void init(double min, double max) {
		
		System.out.println("Randomisation des poids modifiables.");
		Double biasWeight = null;
		for (ILayer layer : layers) {
			
			if (layer.isFirstLayer())
				continue;
			
//			if(impl == ENetworkImplementation.LINKED){
//				
//				List<INode> layersNodes = layer.getLayerNodes();
//				for (INode node : layersNodes) {
//					
//					for (Link link : node.getInputs()){
//						if(link.isWeightModifiable())
//							link.initWeight(min, max);
//					}
//					
//					// initialisation du biais
//					biasWeight = getContext().getNodeBiasWeights()[layer.getLayerId()];
//					if(node.getBiasInput() != null && biasWeight != null){
//						
//						if(biasWeight != 1D){
//							node.getBiasInput().setWeight(getContext().getNodeBiasWeights()[layer.getLayerId()]);
//							node.getBiasInput().setPreviousDeltaWeight(0.0);
//						}else{
//							node.getBiasInput().initWeight(min, max);
//						}
//						
//					}
//					
//					
//				}
//				
//				
//			}else{
//				
//				for(IArea area : layer.getAreas()){
//					if(area.getLinkage().isWeightModifiable()){
//						for(INode node : area.getNodes()){
//							node.setBiasWeightValue(StatUtils.initValue(min, max));
//							node.setBiasPreviousDeltaWeight(0.0);
//						}
//					}
//					
//				}
//				
//			}
			
			for(IArea area : layer.getAreas()){
				if(area.getLinkage().isWeightModifiable()){
					for(INode node : area.getNodes()){
						node.setBiasWeightValue(StatUtils.initValue(min, max));
						node.setBiasPreviousDeltaWeight(0.0);
					}
				}
				
			}

		}
		
		if(impl == ENetworkImplementation.UNLINKED){
			
			for(Map<Identification, Link> entry : Linkage.getLinks().values()){
				for(Link link : entry.values()){
					if(link.isWeightModifiable()){
						link.initWeight(min, max);
					}
				}
			}
		}
		

	}
	
	public void init(double value) {

		
		for (ILayer layer : layers) {
			if (layer.isFirstLayer())
				continue;

			List<INode> layersNodes = layer.getLayerNodes();
			for (INode node : layersNodes) {
				
				for (Link link : node.getInputs())
					if(link.isWeightModifiable())
						link.initWeight(value);
				
				node.getBiasInput().initWeight(value);
			}

		}

	}
	
	public void initBiasWeights(double value) {

		
		for (ILayer layer : layers) {
			
//			if (layer.isFirstLayer())
//				continue;

			List<INode> layersNodes = layer.getLayerNodes();
			for (INode node : layersNodes) {
				
//				for (Link link : node.getInputs())
//					if(link.isWeightModifiable())
//						link.initWeight(value);
				if(getImpl() == null || getImpl() == ENetworkImplementation.LINKED)
					node.getBiasInput().initWeight(value);
				else
					node.setBiasWeightValue(value);
			}

		}

	}

	public void show() {

		Double arg0 = layers.get(0).getLayerNodes().get(0).getInputs().get(0).getValue();
		NN.valinp0.setText(arg0.toString());
		arg0 = layers.get(0).getLayerNodes().get(1).getInputs().get(0).getValue();
		NN.valinp1.setText(arg0.toString());
		arg0 = layers.get(0).getLayerNodes().get(0).getComputedOutput();
		NN.valinp0out.setText(arg0.toString());
		arg0 = layers.get(0).getLayerNodes().get(1).getComputedOutput();
		NN.valinp1out.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getInputs().get(0).getValue() + layers.get(1).getLayerNodes().get(0).getInputs().get(1).getValue();
		NN.valhid0inp.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getInputs().get(0).getValue() + layers.get(1).getLayerNodes().get(0).getInputs().get(1).getValue();
		NN.valhid1inp.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getInputs().get(0).getValue() + layers.get(1).getLayerNodes().get(0).getInputs().get(1).getValue();
		NN.valhid2inp.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getComputedOutput();
		NN.valhid0out.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getComputedOutput();
		NN.valhid1out.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getComputedOutput();
		NN.valhid2out.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getInputs().get(0).getValue() + layers.get(2).getLayerNodes().get(0).getInputs().get(1).getValue()
				+ layers.get(2).getLayerNodes().get(0).getInputs().get(2).getValue();
		NN.valout0inp.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getComputedOutput();
		NN.valout0out.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getError();
		NN.errhid0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getError();
		NN.errhid1.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getError();
		NN.errhid2.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getError();
		NN.errout0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getInputs().get(0).getWeight();
		NN.whid0inp0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getInputs().get(1).getWeight();
		NN.whid0inp1.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getInputs().get(0).getWeight();
		NN.whid1inp0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getInputs().get(1).getWeight();
		NN.whid1inp1.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getInputs().get(0).getWeight();
		NN.whid2inp0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getInputs().get(1).getWeight();
		NN.whid2inp1.setText(arg0.toString());

		if (layers.get(1).getLayerNodes().get(0).getBiasInput() != null) {
			arg0 = layers.get(1).getLayerNodes().get(0).getBiasInput().getWeight();
			NN.wbiashid0.setText(arg0.toString());
		}
		if (layers.get(1).getLayerNodes().get(1).getBiasInput() != null) {
			arg0 = layers.get(1).getLayerNodes().get(1).getBiasInput().getWeight();
			NN.wbiashid1.setText(arg0.toString());
		}
		if (layers.get(1).getLayerNodes().get(2).getBiasInput() != null) {
			arg0 = layers.get(1).getLayerNodes().get(2).getBiasInput().getWeight();
			NN.wbiashid2.setText(arg0.toString());
		}
		if (layers.get(2).getLayerNodes().get(0).getBiasInput() != null) {
			arg0 = layers.get(2).getLayerNodes().get(0).getBiasInput().getWeight();
			NN.wbiasout0.setText(arg0.toString());
		}
		arg0 = layers.get(2).getLayerNodes().get(0).getInputs().get(0).getWeight();
		NN.wout0hid0.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getInputs().get(1).getWeight();
		NN.wout0hid1.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getInputs().get(2).getWeight();
		NN.wout0hid2.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getIdealOutput();
		NN.validealout0.setText(arg0.toString());
		arg0 = layers.get(2).getLayerNodes().get(0).getDerivativeValue();
		NN.valaggout0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(0).getDerivativeValue();
		NN.valagghid0.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(1).getDerivativeValue();
		NN.valagghid1.setText(arg0.toString());
		arg0 = layers.get(1).getLayerNodes().get(2).getDerivativeValue();
		NN.valagghid2.setText(arg0.toString());

	}

	/**
	 * @return
	 */
	public List<INode> getAllNodes() {
		List<INode> allNodes = new ArrayList<INode>();
		for (ILayer layer : layers) {
			allNodes.addAll(layer.getLayerNodes());
		}
		return allNodes;
	}
	
	public List<IArea> getAllAreas() {
		List<IArea> allAreas = new ArrayList<IArea>();
		for (ILayer layer : layers) {
			allAreas.addAll(layer.getAreas());
		}
		return allAreas;
	}
	
	public List<INode> getNodesByType(ENodeType type) {
		List<INode> nodes = new ArrayList<INode>();
		for (ILayer layer : layers) {
			nodes.addAll(layer.getLayerNodes(type));
		}
		return nodes;
	}
	
	/**
	 * 
	 */
	public void disconnectAll(){
		for (ILayer layer : layers) {
			List<INode> layersNodes = layer.getLayerNodes();
			for (INode node : layersNodes) {
				node.disconnect();
			}

		}
	}

	/**
	 * @param entry
	 * @return
	 * @throws Exception
	 */
	public OutputData compute(ListIterator<InputData> dataIter) throws Exception {
		
		int nextIndex = dataIter.nextIndex();
		InputData entries = dataIter.next();
		Iterator<Double> inputIter = entries.getInput().iterator();
		Integer offset = getTimeSeriesOffset();
		Double inputValue = null;

		int dataSetSize = DataSeries.getInstance().getInputDataSet().size();
		int n = 1;

		ListIterator<InputData> tmpItr = null;
		// set input data on input time serie nodes
		List<INode> firstLayerNodes = getFirstLayer().getLayerNodes();
		for (INode node : firstLayerNodes) {

			if (node.getNodeType() == ENodeType.TIMESERIE) {

				if (tmpItr != null && tmpItr.hasNext()) {
					InputData indata = (InputData) tmpItr.next();
					inputIter = indata.getInput().listIterator();
				}

				if (inputIter.hasNext()) {
					inputValue = inputIter.next();
					node.setEntry(inputValue);

					entries.getInput().set(n - 1, inputValue);

					if ((nextIndex + (n * offset)) < dataSetSize - 1)
						tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
					else {
						nextIndex = nextIndex - dataSetSize + 1;
						tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
					}

				} else {
					node.setEntry(0.0D);
				}

			}else{
				inputValue = entries.getInput().get(node.getNodeId());
				node.setEntry(inputValue);
			}
			
			n++;

		}



		return propagation(false);
	}

	/**
	 * @throws Exception
	 */
	public OutputData propagation(boolean playAgain) throws Exception {

		OutputData outputData = null;
		// outputDataList.clear();

		// start of propagation
		getContext().incrementClock();
		
		// one pass forward
		List<ILayer> layers = getLayers();
		Double[] outputValues = null;
		for (ILayer layer : layers) {

			 outputValues = layer.propagate(playAgain);

			// on conserve les sorties pour un eventuel traitement par batch
			if (outputValues != null && layer.isLastLayer()) {
//				outputDataList.addData(outputValues);
				outputData = new OutputData(outputValues);
			}
			
		}
		
		return outputData;

	}
	
	/* (non-Javadoc)
	 * @see RN.INetwork#newLearningCycle()
	 */
	@Override
	public void newLearningCycle(int cycleCount) {
		
		// Initialisation des parametres des noeuds
		for(INode node : getAllNodes())
			node.newLearningCycle(cycleCount);
		
	}

	public String getString() {
		String result = "";
		result += ITester.NEWLINE + "Clock : " + getContext().getClock();
		for (ILayer layer : layers) {
			
			result = layerToString(result, layer);
			
		}
		return result;
	}

	private String layerToString(String result, ILayer layer) {
		
		List<IArea> areas = layer.getAreas();
		int jump = 0;
		
		result += ITester.NEWLINE + layer.toString();
		
		for (IArea area : areas) {
			
			result += ITester.NEWLINE + area.toString();
			
			List<INode> nodes = area.getNodes();
			
			INode node = null;
			if(nodes.size() > 50){
				result += ITester.NEWLINE + ITester.NEWLINE + "        ----------------> Too much nodes to print ("+ nodes.size() + "), we will print the first and last 10th nodes..." + ITester.NEWLINE + ITester.NEWLINE;
				jump = 10;
			}
			for (int id=0; id < nodes.size(); id++) {
				
				if(jump > 0 && id > jump && id < nodes.size() - jump)
					continue;
				else if(jump > 0 && id == jump)
					result += ITester.NEWLINE + "\n\n\n---------------- Jumping to the last 10th nodes... ----------------\n\n\n";
				
				node = nodes.get(id);
				result += ITester.NEWLINE + node.getString();
			}
		}
		return result;
	}
	
	public INode getNode(Identification id){
		return layers.get(id.getLayerId()).getArea(id.getAreaId()).getNode(id.getNodeId());
	}
	
	public INode getNode(int layerId, int areaId, int nodeId){
		
		INode node = layers.get(layerId).getArea(areaId).getNode(nodeId);

		return node;
	}
	
	public IPixelNode getNode(int layerId, int areaId, int x, int y) throws Exception{
		IAreaSquare area = (IAreaSquare) layers.get(layerId).getArea(areaId);
		IPixelNode node = area.getNodeXY(x, y);

		return node;
	}

	public String toString() {

		return this.name;
	}

	public ILayer getFirstLayer() {
		return layers.get(0);
	}

	public ILayer getLastLayer() {
		return layers.get(layers.size() - 1);
	}

	public void setReccurent(Integer layerRecurrentId) {
		this.recurrentLayerId = layerRecurrentId;
	}

	public Integer getRecurrentLayerId() {
		return recurrentLayerId;
	}

	public boolean isRecurrent() {
		return recurrentLayerId != null;
	}

	public Integer getTimeSeriesOffset() {
		return timeSeriesOffset;
	}

	public void setTimeSeriesOffset(Integer timeSeriesOffset) {
		this.timeSeriesOffset = timeSeriesOffset;
	}

	@Override
	public void setRecurrentNodesLinked(Boolean lateralLinkRecurrentNodes) {
		this.recurrentNodesLinked = lateralLinkRecurrentNodes;
	}

	@Override
	public Boolean isRecurrentNodesLinked() {
		return recurrentNodesLinked;
	}
	
	@Override
	public double getAbsoluteError() {
		return absoluteError;
	}

	@Override
	public void setAbsoluteError(double absoluteError) {
		this.absoluteError = absoluteError;
	}

	@Override
	public INetwork deepCopy(int generationCount) {
		String name = "";
		if(getName().startsWith("G")){
			name = "G" + generationCount + getName().substring(getName().indexOf(Genetic.GENE_SEPARATOR)) ;
		}else{
			name = "G" + generationCount + Genetic.GENE_SEPARATOR + getName();
		}
		Network copy_network = Network.getInstance();
		copy_network.setName(name);
		List<ILayer> copy_layers = new ArrayList<ILayer>(layers);
		Collections.copy(copy_layers, layers);
		
		OutputDataList copy_outputDataList = new OutputDataList();
		copy_network.setLayers(copy_layers);
		copy_network.setOutputDataList(copy_outputDataList);
		copy_network.setRecurrentNodesLinked(new Boolean(recurrentNodesLinked));
		copy_network.setTimeSeriesOffset(new Integer(timeSeriesOffset));
		
		int idx = 0;
		for(ILayer layer : layers){
			layer.setNetwork(copy_network);
			copy_layers.set(idx++, layer.deepCopy());
		}
		
		return copy_network;
	}
	
	public String geneticCodec() {

		String geneticCode = "";
		
		geneticCode += "I(" + this.getFirstLayer().getNodeCountMinusRecurrentOnes() + ")";
		geneticCode += Genetic.CODE_SEPARATOR;

		if (this.getLayer(1).getNodeCount() > 0)
			geneticCode += "H0(" + this.getLayer(1).getNodeCount() + ")";
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 1)
			geneticCode += "H1(" + this.getLayer(2).getNodeCount() + ")";
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 2)
			geneticCode += "H1(" + this.getLayer(3).getNodeCount() + ")";
		geneticCode += Genetic.CODE_SEPARATOR;		
		geneticCode += "O(" + this.getLastLayer().getNodeCount() + ")";
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLastLayer().isLayerReccurent())
			geneticCode += "Ro" ;
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayer(1).isLayerReccurent())
			geneticCode +=  "Rh1" ;
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 1 && this.getLayer(2).isLayerReccurent())
			geneticCode += "Rh2" ;
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 2 && this.getLayer(3).isLayerReccurent())
			geneticCode +=  "Rh3" ;
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.isRecurrentNodesLinked())
			geneticCode += "-R-" ;
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayer(1).getFunction() != null)
			geneticCode +=  "Xh0" + this.getLayer(1).getFunction().name().charAt(0);
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 1)
			geneticCode += "Xh1" + this.getLayer(2).getFunction().name().charAt(0);
		geneticCode += Genetic.CODE_SEPARATOR;
		if (this.getLayers().size() - 2 > 2)
			geneticCode += "Xh2" + this.getLayer(3).getFunction().name().charAt(0);
		geneticCode += Genetic.CODE_SEPARATOR;
		
		geneticCode +=  "Xo" + this.getLastLayer().getFunction().name().charAt(0);
		geneticCode += Genetic.GENE_SEPARATOR;
		
		return geneticCode;
	}

	public OutputDataList getOutputDataList() {
		return outputDataList;
	}

	public void setOutputDataList(OutputDataList outputDataList) {
		this.outputDataList = outputDataList;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void setRecurrentLayerId(Integer recurrentLayerId) {
		this.recurrentLayerId = recurrentLayerId;
	}
	
	public void appendName(String appended){
		this.name += appended;
	}

	public ENetworkImplementation getImpl() {
		return impl;
	}

	public void setImpl(ENetworkImplementation impl) {
		this.impl = impl;
	}






}
