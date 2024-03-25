package RN.algotrainings;

import java.io.IOException;
import java.util.List;
import java.util.ListIterator;

import javafx.collections.FXCollections;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import RN.DataSeries;
import RN.TestLSTMNetwork;
import RN.ViewerFX;
import RN.dataset.InputData;
import RN.dataset.OutputData;
import dmonner.xlbp.Network;
import dmonner.xlbp.compound.InputCompound;
import dmonner.xlbp.compound.XEntropyTargetCompound;

/**
 * @author Eric Marchand
 *
 */
public class LSTMTrainer implements ITrainer {

	private static LSTMTrainer instance;
	
	private double errorRate;
	
	final public static double delta = 0.0;
	
	final public static double delay = 0.0;	
	
	static int trainCycleAbsolute = 0;
	
	public static LSTMTrainer getInstance() {
		if (instance == null) {
			instance = new LSTMTrainer();
		}

		return instance;
	}
	
	@Override
	public void train() throws Exception {
		launchTrain();
	}
	
	@Override
	public void launchTrain() throws IOException {
		launchTrain(true);
	}
	

	@Override
	public void launchTrain(boolean verbose) throws IOException {

		DataSeries dataSeries = DataSeries.getInstance();
		InputCompound in = TestLSTMNetwork.getInstance().getIn();
		XEntropyTargetCompound out = TestLSTMNetwork.getInstance().getOut();
		Network net = (Network) TestLSTMNetwork.getNet();
		
		
		LineChart<Number, Number> sc = ViewerFX.lineChart;
		if (sc.getData() == null)
			sc.setData(FXCollections.<XYChart.Series<Number, Number>> observableArrayList());
		LineChart.Series<Number, Number> series = new LineChart.Series<Number, Number>();
		series.setName("Train " + (sc.getData().size() + 1));

		
		int trainCycle = 0;
		double absoluteError = 0.0D;
		double squaredError = 0.0D;
		do {
			absoluteError = 0.0D;
			
			// prochain jeu de test
			for (InputData entries : dataSeries.getInputDataSet()) {

				// init de l'erreur totale sur la couche de sortie
				setErrorRate(0.0);
				
				// Clears the network, setting all units to their default
				// activation levels
				// net.clear();

				// for(int i = 0; i < insize; i++)
				// input[i] = (float) Math.random();

				// Impose an input vector on the units of the input layer
				in.setInput(entries.getInputArray());

				// Activate the layers of the network, caching information
				// necessary for weight updates;
				// If we're not updating weights on this trial, activateTest()
				// is faster.
				net.activateTrain();

				// If we're going to update weights, we need to update each
				// unit's eligibility.
				net.updateEligibilities();

				// for(int i = 0; i < outsize; i++)
				// target[i] = (float) Math.random();

				// Set the target vector that the output layer should be trying
				// to obtain, for training.
				out.setTarget(entries.getIdealArray());

				// Propagate unit responsibilities backwards from the output
				// according to LSTM-g
				net.updateResponsibilities();

				// Update the weights according to LSTM-g
				net.updateWeights();

				for(int idx = 0 ; idx < entries.getIdealArray().length; idx++){
					squaredError =  Math.pow(entries.getIdealArray()[idx] - out.getOutput().getActivations()[idx], 2.0D) / 2.0D;
					errorRate += squaredError;
					absoluteError += squaredError;
				}
//				if(trainCycleAbsolute % 100 == 0)
//					System.out.println("X#" + trainCycleAbsolute + " Error:" + absoluteError + " ideal:" + entries.getIdealArray()[0] + " res:"+ out.getOutput().getActivations()[0]);
				
			}
			
			if(verbose && trainCycleAbsolute % 100 == 0)
				System.out.println("Stage #" + trainCycleAbsolute + " Error:" + absoluteError);
			
			series.getData().add(new ScatterChart.Data<Number, Number>(trainCycle++, absoluteError));
			trainCycleAbsolute++;
		} while (trainCycle < 500);

		sc.getData().add(series);
		// System.out.println(net.toString("NICXW"));

	}

	
	public double getDelta() {
		return delta;
	}

	public double getDelay() {
		return delay;
	}
	

	@Override
	public double getErrorRate() {
		// TODO Auto-generated method stub
		return this.errorRate;
	}
	
	private void setErrorRate(double d) {
		this.errorRate = d;
	}	

	@Override
	public void nextTrainInputValues() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void feedForward() {
		// TODO Auto-generated method stub
		
	}

	public void computeDeltaWeights() {
		// TODO Auto-generated method stub
		
	}

	public void updateAllWeights() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setLearningRate(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setAlphaDeltaWeight(double alphaDeltaWeight) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public double getLearningRate() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double getAlphaDeltaWeight() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setInputDataSetIterator(ListIterator<InputData> inputDataSetIterator) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public List<Data<Number, Number>> getErrorLevelLines() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setErrorLevelLines(List<Data<Number, Number>> lines) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initTrainer() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public InputData getCurrentEntry() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public OutputData getCurrentOutputData() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setBreakTraining(boolean breakTraining) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getAbsoluteError() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setMaxTrainingCycles(int maxTrainingCycles) {
		// TODO Auto-generated method stub
		
	}



}
