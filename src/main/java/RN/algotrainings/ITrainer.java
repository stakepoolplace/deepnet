package RN.algotrainings;

import java.util.List;
import java.util.ListIterator;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import javafx.scene.chart.LineChart;
import javafx.scene.control.TextArea;


/**
 * @author Eric Marchand
 *
 */
public interface ITrainer {

	void train() throws Exception;
	
	double getErrorRate();

	public abstract void nextTrainInputValues();

	void launchTrain() throws Exception;
	
	void launchTrain(int nbTrainingCycles) throws Exception;
	
	void launchTrain(boolean verbose, TextArea console) throws Exception;

	public abstract double getDelay();
	
	public abstract double getDelta();

	public abstract double getLearningRate();

	public abstract void setLearningRate(double learningRate);

	public abstract double getAlphaDeltaWeight();

	public abstract void setAlphaDeltaWeight(double alphaDeltaWeight);

	public abstract void setMomentum(double alphaDeltaWeight);

	void feedForward() throws Exception;
	
	boolean backPropagateError() throws Exception;

	void setInputDataSetIterator(ListIterator<InputData> inputDataSetIterator);
	
	List<LineChart.Data<Number, Number>> getErrorLevelLines();
	
	void setErrorLevelLines(List<LineChart.Data<Number, Number>> lines);

	void initTrainer();

	InputData getCurrentEntry();

	OutputData getCurrentOutputData();
	
	void setBreakTraining(boolean breakTraining);
	
	double getAbsoluteError();

	void setMaxTrainingCycles(int maxTrainingCycles);
	
	void setCurrentOutputData(OutputData currentOutputData);

}
