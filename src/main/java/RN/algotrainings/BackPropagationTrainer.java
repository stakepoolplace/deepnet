package RN.algotrainings;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import RN.DataSeries;
import RN.ENetworkImplementation;
import RN.ILayer;
import RN.INetwork;
import RN.TestNetwork;
import RN.algoactivations.EActivation;
import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.strategy.EStrategy;
import RN.strategy.Strategy;
import RN.strategy.StrategyFactory;
import javafx.scene.chart.LineChart;
import javafx.scene.control.TextArea;

/**
 * @author Eric Marchand
 *
 */
public class BackPropagationTrainer implements ITrainer {

	private static int maxTrainingCycles = 500;

	protected double errorRate;
	private double absoluteError = 0.0D;

	private volatile boolean breakTraining = false;

	private ListIterator<InputData> inputDataSetIterator = null;
	private List<LineChart.Data<Number, Number>> errorLevel = new ArrayList<LineChart.Data<Number, Number>>(maxTrainingCycles);

	private static int trainCycleAbsolute = 0;

	protected double learningRate = 0.5D;
	protected double alphaDeltaWeight = 0.0D;
	private int meanPeriodCount = 0;

	private double delta = 0.05;
	private double delay = 0.1;
	
	private InputData currentEntry = null;
	private OutputData currentOutputData = null;

	public BackPropagationTrainer() {
		errorRate = 0;
	}

	public void initTrainer() {
		trainCycleAbsolute = 0;
		errorLevel.clear();
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
				
				long start = System.currentTimeMillis();
				
				absoluteError = 0.0D;
				getNetwork().newLearningCycle(trainCycleAbsolute);
				for (int ind = 1; ind <= samplesNb; ind++) {
					train();
					absoluteError += getErrorRate();
				}
				absoluteError =  Math.sqrt(absoluteError / samplesNb);
				getNetwork().setAbsoluteError(absoluteError);

				// Sampling 1/10 de l'affichage de l'erreur
				if(trainCycle % (maxTrainingCycles / 100) == 0)
					errorLevel.add(new LineChart.Data<Number, Number>(trainCycleAbsolute, absoluteError));
				
//				if(ViewerFX.growingHiddens.isSelected() && (trainCycle % 20 == 0) && sigmaAbsoluteError > 0.0D && ((sigmaAbsoluteError / (trainCycle + 1)) <= absoluteError * 1.1D))
//					strategy.apply();
//				
				long stop = System.currentTimeMillis();
				if (verbose) {
					if(console != null)
						console.appendText("Stage #" + trainCycleAbsolute + "    Error: " + absoluteError + "    Error mean: " + (sigmaAbsoluteError / (trainCycle + 1)) + "    Duration: "+ (stop-start)/1000 + " second(s)"+ "\n");
					else
						System.out.println("Stage #" + trainCycleAbsolute + "    Error: " + absoluteError + "    Error mean: " + (sigmaAbsoluteError / (trainCycle + 1)) + "    Duration: "+ (stop-start)/1000 + " second(s)");
				}
				
				trainCycleAbsolute++;
				trainCycle++;
				sigmaAbsoluteError += absoluteError;
				// }while(train.getErrorRate() > 0.00001);
			} while (!breakTraining && trainCycle < maxTrainingCycles && absoluteError > 0.001);

		}
	}

	public void train() throws Exception {

			boolean playAgain = iterateHasNext();
			
			// feed forward
			currentOutputData = getNetwork().propagation(playAgain);
	
			backPropagateError();
			

	}

	private boolean iterateHasNext() {
		
		DataSeries data = DataSeries.getInstance();
		ILayer firstLayer = getNetwork().getFirstLayer();
		ILayer lastLayer = getNetwork().getLastLayer();
		Iterator<INode> outputNodeIter = null;
		Iterator<Double> inputIter = null;
		
		boolean isSameTrainingInputs = true;
		

		// init de l'erreur totale sur la couche de sortie
		setErrorRate(0.0);

		// si l'iterator n'existe pas on le cree
		if (inputDataSetIterator == null) {
			inputDataSetIterator = data.getInputDataSet().listIterator(0);
		}
		
		if (!inputDataSetIterator.hasNext()) {
			
			if(data.getInputDataSet().size() == 1){
				return true;
			}
			
			inputDataSetIterator = data.getInputDataSet().listIterator(0);
		}

		// prochain jeu de test
		if (inputDataSetIterator.hasNext()) {
			
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

				}else if(node.getNodeType() == ENodeType.REGULAR || node.getNodeType() == ENodeType.PIXEL){
					
					inputValue = currentEntry.getInput().get(node.getNodeId());
					
					if(!inputValue.equals(node.getEntry()))
						isSameTrainingInputs = false;
					
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
		
		return isSameTrainingInputs;
	}

	public void nextTrainInputValues() {

		DataSeries data = DataSeries.getInstance();
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

				}else if(node.getNodeType() == ENodeType.REGULAR  ||
						node.getNodeType() == ENodeType.PIXEL){
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


	public void initError() {
		for (INode node : getNetwork().getAllNodes()) {
			node.setDerivatedError(0.0);
		}
	}

	public void feedForward() {
		// feed forward
		try {
			getNetwork().propagation(false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	public boolean backPropagateError() throws Exception {

		List<ILayer> layers = getNetwork().getLayers();
		int layerCount = layers.size();
		ILayer layer = null;

		setErrorRate(0.0D);

		for (int ind = layerCount - 1; ind >= 0; ind--) {
			
			layer = layers.get(ind);
			layer.setLayerError(0.0D);
			
			for (final INode node : layer.getLayerNodes(ENodeType.REGULAR, ENodeType.PIXEL)) {

				if(!node.getArea().getLinkage().isWeightModifiable())
					continue;
				
				double derivatedError = 0.0D;

				// la valeur dérivée à été calculée lors du feedforward
				
				if (layer.isLastLayer()) {
					
					
					// on defini l'erreur totale de la couche de sortie
					derivatedError = node.getError() * node.getDerivativeValue();
					
					// on se dirige vers les minimum locaux
					// Backpropagation in-line (not batch), algo LMS (Least Mean Squared)
					errorRate += Math.pow(node.getError(), 2.0D);
					

				} else {

					// somme pondérés du produit des poids des noeuds reliés sur
					// la couche précédente et de l'erreur du noeud reliés sur
					// la couche precedente
					if(getNetwork().getImpl() == ENetworkImplementation.LINKED){
						
						for (Link link : node.getOutputs()) {
							if(link != null && (link.getType() == ELinkType.REGULAR || link.getType() == ELinkType.SHARED)){
								derivatedError += link.getWeight() * link.getTargetNode().getDerivatedError();
							}
						}
						
					}else{
						
						derivatedError = node.getDerivatedErrorSum();
						
					}
					
					node.setError(derivatedError);
					
					derivatedError *= node.getDerivativeValue();


				}

				// on defini l'erreur aggregée sur le node
				// node.setDerivatedError(node.getDerivatedError() + derivatedError);
				// la backpropagation in-line ne cumule pas les erreurs sur les
				// differents jeux de test
				node.setDerivatedError(derivatedError);
				
				layer.setLayerError(layer.getLayerError() + derivatedError);
				
				node.updateWeights(learningRate, alphaDeltaWeight);
					
			}

			// on ne calcul pas le delta des poids si la sortie et egale au
			// resultat désiré, on sort dés la première itération
			if (layer.isLastLayer() && getErrorRate() == 0.0D)
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
	
	public void setMomentum(double alphaDeltaWeight) {
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
		return errorLevel;
	}

	public void setErrorLevelLines(List<LineChart.Data<Number, Number>> errorLevel) {
		this.errorLevel = errorLevel;
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
		BackPropagationTrainer.maxTrainingCycles = maxTrainingCycles;
	}




	
}
