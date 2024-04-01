package RN.algotrainings;

import java.util.List;
import java.util.ListIterator;

import javafx.scene.chart.LineChart;
import RN.dataset.InputData;
import RN.dataset.OutputData;


/**
 * @author Eric Marchand
 *
 */
public interface ITrain {

	void train() throws Exception;

	double getErrorRate();

	public abstract void nextTrainInputValues();

	public abstract void launchTrain() throws Exception;

	public abstract double getDelay();
	
	public abstract double getDelta();

	public abstract double getLearningRate();

	public abstract void setLearningRate(double learningRate);

	public abstract double getAlphaDeltaWeight();

	public abstract void setAlphaDeltaWeight(double alphaDeltaWeight);

	void feedForward() throws Exception;

	void setInputDataSetIterator(ListIterator<InputData> inputDataSetIterator);
	
	List<LineChart.Data<Number, Number>> getLines();
	
	void setLines(List<LineChart.Data<Number, Number>> lines);

	void initTrainer();

	InputData getCurrentEntry();

	OutputData getCurrentOutputData();
	
	void setBreakTraining(boolean breakTraining);
	
	double getAbsoluteError();

	void setMaxTrainingCycles(int maxTrainingCycles);

}
