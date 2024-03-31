package RN.algotrainings;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import RN.DataSeries;
import RN.ILayer;
import RN.INetwork;
import RN.TestNetwork;
import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.LSTMNode;
import RN.strategy.EStrategy;
import RN.strategy.Strategy;
import RN.strategy.StrategyFactory;
import javafx.scene.chart.LineChart;
import javafx.scene.control.TextArea;

/**
 * @author Eric Marchand
 *
 */
public class LSTMBackwardPassTraining implements ITrainer {

	private static int maxTrainingCycles = 500;

	private double errorRate;
	private double absoluteError = 0.0D;

	private volatile boolean breakTraining = false;

	private ListIterator<InputData> inputDataSetIterator = null;
	private List<LineChart.Data<Number, Number>> lines = new ArrayList<LineChart.Data<Number, Number>>(maxTrainingCycles);

	private static int trainCycleAbsolute = 0;

	private double learningRate = 1.0D;
	private double alphaDeltaWeight = 0.0D;
	private int meanPeriodCount = 0;

	private double delta = 0.05;
	private double delay = 0.1;
	
	private InputData currentEntry = null;
	private OutputData currentOutputData = null;

	public LSTMBackwardPassTraining() {
		errorRate = 0;
	}

	public void initTrainer() {
		trainCycleAbsolute = 0;
		lines.clear();
	}

	@Override
	public void launchTrain() throws Exception {
		launchTrain(true, null);
	}
	
	@Override
	public void launchTrain(int nbTrainingCycles) throws Exception {
		setMaxTrainingCycles(nbTrainingCycles);
		launchTrain(true, null);
		
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.ITester#lauchTrain(java.lang.Long, java.lang.Long)
	 */
	@Override
	public void launchTrain(boolean verbose, TextArea console) throws Exception {

		int samplesNb = DataSeries.getInstance().getInputDataSet().size();
		breakTraining = false;
		
		if (samplesNb > 0) {

			int trainCycle = 0;
			absoluteError = 0.0D;
			double sigmaAbsoluteError = 0.0D;
			Strategy strategy = StrategyFactory.create(getNetwork(), EStrategy.AUTO_GROWING_HIDDENS);
			do {
				absoluteError = 0.0D;
				for (int ind = 1; ind <= samplesNb; ind++) {
					train();
					absoluteError += getErrorRate();
				}
				absoluteError =  Math.sqrt(absoluteError / samplesNb);
				getNetwork().setAbsoluteError(absoluteError);

				// Sampling 1/10 de l'affichage de l'erreur
				if(trainCycle % (maxTrainingCycles / 100) == 0)
					lines.add(new LineChart.Data<Number, Number>(trainCycleAbsolute, absoluteError));
				
//				if(ViewerFX.growingHiddens.isSelected() && (trainCycle % 20 == 0) && sigmaAbsoluteError > 0.0D && ((sigmaAbsoluteError / (trainCycle + 1)) <= absoluteError * 1.1D))
//					strategy.apply();
//				
				
				if (verbose && (trainCycle % 100 == 0)) {
					if(console != null)
						console.appendText("Stage #" + trainCycleAbsolute + " Error:" + absoluteError + "  Error mean:" + (sigmaAbsoluteError / (trainCycle + 1)) + "\n");
					else
						System.out.println("Stage #" + trainCycleAbsolute + " Error:" + absoluteError + "  Error mean:" + (sigmaAbsoluteError / (trainCycle + 1)));
				}
				
				
				
				trainCycleAbsolute++;
				trainCycle++;
				sigmaAbsoluteError += absoluteError;
				// }while(train.getErrorRate() > 0.00001);
			} while (!breakTraining && trainCycle < maxTrainingCycles && absoluteError > 0.001);

		}
	}

	public void train() throws Exception {

		if(iterateHasNext()){
		
			// feed forward
			currentOutputData = getNetwork().propagation(true);
	
			backPropagateError();
			
		}

	}

	private boolean iterateHasNext() {
		DataSeries data = DataSeries.getInstance();
		ILayer firstLayer = getNetwork().getFirstLayer();
		ILayer lastLayer = getNetwork().getLastLayer();
		Iterator<INode> outputNodeIter = null;
		Iterator<Double> inputIter = null;
		boolean hasNext = false;
		

		// init de l'erreur totale sur la couche de sortie
		setErrorRate(0.0);

		// si l'iterator n'existe pas on le cree
		if (inputDataSetIterator == null) {
			inputDataSetIterator = data.getInputDataSet().listIterator(0);
		}
		if (!inputDataSetIterator.hasNext()) {
			// return;
			inputDataSetIterator = data.getInputDataSet().listIterator(0);
		}

		// prochain jeu de test
		if (inputDataSetIterator.hasNext()) {
			hasNext = true;
			int nextIndex = inputDataSetIterator.nextIndex();
			currentEntry = inputDataSetIterator.next();

			inputIter = currentEntry.getInput().iterator();
			Integer offset = getNetwork().getTimeSeriesOffset();
			Double inputValue = null;
			int dataSetSize = data.getInputDataSet().size();
			int n = 1;

			ListIterator<InputData> tmpItr = null;

			List<INode> firstLayerNodes = firstLayer.getLayerNodes();
			for (INode node : firstLayerNodes) {

				if (node.getNodeType() == ENodeType.TIMESERIE) {

					if (tmpItr != null && tmpItr.hasNext()) {
						InputData indata = (InputData) tmpItr.next();
						inputIter = indata.getInput().listIterator();
					}

					if (inputIter.hasNext()) {
						inputValue = inputIter.next();
						node.setEntry(inputValue);

						if ((nextIndex + (n * offset)) < dataSetSize - 1)
							tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
						else {
							nextIndex = nextIndex - dataSetSize + 1;
							tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
						}

					} else {
						node.setEntry(0.0D);
					}

				}else if(node.getNodeType() == ENodeType.REGULAR){
					inputValue = currentEntry.getInput().get(node.getNodeId());
					node.setEntry(inputValue);
				}else{
					continue;
				}
				
				n++;

			}

			// for sur une sortie ideale du jeu
			INode node = null;
			outputNodeIter = lastLayer.getLayerNodes(ENodeType.REGULAR).iterator();
			for (Double ideal : currentEntry.getIdeal()) {
				if (outputNodeIter.hasNext()) {
					node = outputNodeIter.next();
					node.setIdealOutput(ideal);
				}
			}

		}
		
		return hasNext;
	}

	public void nextTrainInputValues() {

		DataSeries data = DataSeries.getInstance();
		ILayer firstLayer = getNetwork().getFirstLayer();
		ILayer lastLayer = getNetwork().getLastLayer();
		Iterator<INode> outputNodeIter = null;

		// init de l'erreur totale sur la couche de sortie
		setErrorRate(0.0);

		if (inputDataSetIterator == null) {
			inputDataSetIterator = data.getInputDataSet().listIterator();
		}
		if (!inputDataSetIterator.hasNext()) {
			inputDataSetIterator = data.getInputDataSet().listIterator();
		}

		// prochain jeu de test
		if (inputDataSetIterator.hasNext()) {
			int nextIndex = inputDataSetIterator.nextIndex();
			InputData entries = inputDataSetIterator.next();

			Iterator<Double> inputIter = entries.getInput().iterator();
			Integer offset = getNetwork().getTimeSeriesOffset();
			Double inputValue = null;
			
			int dataSetSize = data.getInputDataSet().size();
			int n = 1;

			ListIterator<InputData> tmpItr = null;

			List<INode> firstLayerNodes = getNetwork().getFirstLayer().getLayerNodes();
			for (INode node : firstLayerNodes) {

				if (node.getNodeType() == ENodeType.TIMESERIE) {

					if (tmpItr != null && tmpItr.hasNext()) {
						InputData indata = (InputData) tmpItr.next();
						inputIter = indata.getInput().listIterator();
					}

					if (inputIter.hasNext()) {
						inputValue = inputIter.next();
						node.setEntry(inputValue);

						if ((nextIndex + (n * offset)) < dataSetSize - 1)
							tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
						else {
							nextIndex = nextIndex - dataSetSize + 1;
							tmpItr = DataSeries.getInstance().getInputDataSet().listIterator(nextIndex + (n * offset));
						}

					} else {
						node.setEntry(0.0D);
					}

				}else if(node.getNodeType() == ENodeType.REGULAR){
					inputValue = entries.getInput().get(node.getNodeId());
					node.setEntry(inputValue);
				}else{
					continue;
				}
				
				n++;

			}


			// for sur une sortie ideale du jeu
			INode node = null;
			outputNodeIter = lastLayer.getLayerNodes(ENodeType.REGULAR).iterator();
			for (double ideal : entries.getIdeal()) {
				if (outputNodeIter.hasNext()) {
					node = outputNodeIter.next();
					node.setIdealOutput(ideal);
				}
			}

		}

		initError();

	}

	//
	// public void nextTrainInputValues(InputData entries) {
	//
	// // set input values
	// Layer firstLayer = getNetwork().getFirstLayer();
	// Layer lastLayer = getNetwork().getLastLayer();
	// Iterator<Node> outputNodeIter = null;
	//
	// // init de l'erreur totale sur la couche de sortie
	// setErrorRate(0.0);
	//
	// // prochain jeu de test
	// if (entries != null) {
	//
	// outputNodeIter = lastLayer.getLayerNodes(ENodeType.REGULAR).iterator();
	//
	// List<Node> firstLayerNodes = firstLayer.getLayerNodes(ENodeType.REGULAR);
	// Iterator<Double> inputIter = entries.getInput().iterator();
	//
	// Double inputValue;
	// for (Node node : firstLayerNodes) {
	//
	// if (inputIter.hasNext()) {
	// inputValue = inputIter.next();
	// node.getInputs().get(0).setValue(inputValue);
	// // pas de fonction de transfert pour la couche d'entree
	// // node.getOutput().setValue(entry);
	//
	// }
	//
	// }
	//
	// // for sur une sortie ideale du jeu
	// Node node = null;
	// for (double ideal : entries.getIdeal()) {
	// if (outputNodeIter.hasNext()) {
	// node = outputNodeIter.next();
	// node.setIdealOutput(ideal);
	// }
	// }
	//
	// }
	//
	// initError();
	//
	// }

	public void initError() {
		for (INode node : getNetwork().getAllNodes()) {
			node.setDerivatedError(0.0);
		}
	}

	public void feedForward() {
		// feed forward
		try {
			getNetwork().propagation(true);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	private boolean backPropagateError() throws Exception {

		List<ILayer> layers = getNetwork().getLayers();
		int layerCount = layers.size();
		ILayer layer = null;

		setErrorRate(0.0D);

		for (int ind = layerCount - 1; ind >= 0; ind--) {
			layer = layers.get(ind);
			layer.setLayerError(0.0D);
			for (final INode node : layer.getLayerNodes()) {
				
				if(node.getNodeType() == ENodeType.REGULAR){

				double derivativeGx = 0.0D;
				double derivatedError = 0.0D;

				// la valeur dérivée à été calculée lors du feedforward, on la
				// récupere.
				derivativeGx = node.getDerivativeValue();

				if (layer.isLastLayer()) {
					// on defini l'erreur totale de la couche de sortie
					derivatedError = node.getError() * derivativeGx;
					// on se dirige vers les minimum locaux : TODO a modifier
					// backpropagation in-line (not batch), algo LMS (Least Mean
					// Squared)
					errorRate += Math.pow(node.getError(), 2.0D);

				} else {

					// somme pondérés du produit des poids des noeuds reliés sur
					// la couche précédente et de l'erreur du noeud reliés sur
					// la couche precedente
					for (Link link : node.getOutputs()) {
						if(link.getType() == ELinkType.REGULAR){
							derivatedError += link.getWeight() * link.getTargetNode().getDerivatedError();
						}
					}
					derivatedError *= derivativeGx;

				}

				// on defini l'erreur aggregé sur le node
				// node.setDerivatedError(node.getDerivatedError() + derivatedError);
				// la backpropagation in-line ne cumule pas les erreurs sur les
				// differents jeux de test
				node.setDerivatedError(derivatedError);
				layer.setLayerError(layer.getLayerError() + derivatedError);
				
				
				node.updateWeights(learningRate, alphaDeltaWeight);
				
				} else if(node.getNodeType() == ENodeType.LSTM){

					LSTMNode lstm = (LSTMNode) node;
					double derivativeGx = 0.0D;
					double derivatedError = 0.0D;

					for (Link link : node.getOutputs()) {
						if(link.getType() == ELinkType.REGULAR){
							derivatedError = link.getWeight() * link.getTargetNode().getDerivatedError();
						}
						
						INode piOutputNode = link.getSourceNode();
						derivatedError *= node.getDerivativeValue();
						
						
					}
					
					
//					lstm.
					
					
					
				}
					
			}

			// on ne calcul pas le delta des poids si la sortie et egale au
			// resultat désiré
			if (layer.isLastLayer() && getErrorRate() == 0.0)
				return false;

		}

		return true;

	}


	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.ITester#getLearningRate()
	 */
	@Override
	public double getLearningRate() {
		return learningRate;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.ITester#getAlphaDeltaWeight()
	 */
	@Override
	public double getAlphaDeltaWeight() {
		return alphaDeltaWeight;
	}

	@Override
	public double getErrorRate() {
		return this.errorRate;
	}

	public void setErrorRate(double errorRate) {
		this.errorRate = errorRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public void setAlphaDeltaWeight(double alphaDeltaWeight) {
		this.alphaDeltaWeight = alphaDeltaWeight;
	}

	@Override
	public double getDelay() {
		return delay;
	}

	@Override
	public double getDelta() {
		return delta;
	}

	public INetwork getNetwork() {
		return TestNetwork.network;
	}

	public Iterator<InputData> getInputDataSetIterator() {
		return inputDataSetIterator;
	}

	public void setInputDataSetIterator(ListIterator<InputData> inputDataSetIterator) {
		this.inputDataSetIterator = inputDataSetIterator;
	}

	public List<LineChart.Data<Number, Number>> getErrorLevelLines() {
		return lines;
	}

	public void setErrorLevelLines(List<LineChart.Data<Number, Number>> lines) {
		this.lines = lines;
	}

	public InputData getCurrentEntry() {
		return currentEntry;
	}

	public void setCurrentEntry(InputData currentEntry) {
		this.currentEntry = currentEntry;
	}

	public OutputData getCurrentOutputData() {
		return currentOutputData;
	}

	public void setCurrentOutputData(OutputData currentOutputData) {
		this.currentOutputData = currentOutputData;
	}

	public boolean isBreakTraining() {
		return breakTraining;
	}

	public void setBreakTraining(boolean breakTraining) {
		this.breakTraining = breakTraining;
	}

	public double getAbsoluteError() {
		return absoluteError;
	}

	public void setAbsoluteError(double absoluteError) {
		this.absoluteError = absoluteError;
	}

	public int getMaxTrainingCycles() {
		return maxTrainingCycles;
	}

	public void setMaxTrainingCycles(int maxTrainingCycles) {
		LSTMBackwardPassTraining.maxTrainingCycles = maxTrainingCycles;
	}

	public void setMomentum(double alphaDeltaWeight) {
		this.alphaDeltaWeight = alphaDeltaWeight;
	}


	
}
