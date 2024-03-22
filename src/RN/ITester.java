package RN;

import java.util.List;

import javafx.scene.chart.LineChart;

public interface ITester {

	public final static String NEWLINE = System.getProperty("line.separator");

	public abstract INetwork createNetwork(String networkName);
	
	public abstract void launchRealCompute() throws Exception;

	public abstract INetwork initWeights(double min, double max);
	
	public abstract int getInputsCount();
	
	public abstract int getOutputsCount();
	
	public abstract INetwork getNetwork();

	void setInputsCount(int insize);

	void setOutputsCount(int outsize);
	
	List<LineChart.Series<Number, Number>> getSeriesInList();

	void setSeriesInList(List<LineChart.Series<Number, Number>> seriesInList);

	List<LineChart.Series<Number, Number>> getSeriesOutList();

	void setSeriesOutList(List<LineChart.Series<Number, Number>> seriesOutList);

	List<LineChart.Series<Number, Number>> getSeriesIdealList();

	void setSeriesIdealList(List<LineChart.Series<Number, Number>> seriesIdealList);


	void setNetwork(INetwork net);
	
	String getFilePath();
	
	Integer getTimeSeriesOffset();

	void setTrainingVectorNumber(Integer lastRowNum);

	Integer getLayerHidden0NodesCount();

	void setLayerHidden0NodesCount(Integer layerHidden0NodesCount);
	
	Integer getLayerHidden1NodesCount();

	void setLayerHidden1NodesCount(Integer layerHidden1NodesCount);

	Integer getLayerHidden2NodesCount();

	void setLayerHidden2NodesCount(Integer layerHidden2NodesCount);

	Integer getLayerOutNodesCount();

	void setLayerOutNodesCount(Integer layerOutNodesCount);

	String getLayerOutActivation();

	void setLayerOutActivation(String layerOutActivation);

	Boolean getLayerOutRecurrent();

	void setLayerOutRecurrent(Boolean layerOutRecurrent);

	String getLayerHidden0Activation();

	void setLayerHidden0Activation(String layerHidden0Activation);

	Boolean getLayerHidden0Recurrent();

	void setLayerHidden0Recurrent(Boolean layerHidden0Recurrent);

	String getLayerHidden1Activation();

	void setLayerHidden1Activation(String layerHidden1Activation);

	Boolean getLayerHidden1Recurrent();

	void setLayerHidden1Recurrent(Boolean layerHidden1Recurrent);

	String getLayerHidden2Activation();

	void setLayerHidden2Activation(String layerHidden2Activation);

	Boolean getLayerHidden2Recurrent();

	void setLayerHidden2Recurrent(Boolean layerHidden2Recurrent);

	Integer getLayerInNodesCount();

	void setLayerInNodesCount(Integer layerInNodesCount);

	String getLayerInActivation();

	void setLayerInActivation(String layerInActivation);

	int getOptimizedNumHiddens();

	void setOptimizedNumHiddens(int optimizedNumHiddens);

	INetwork createXLSNetwork(String networkName, NetworkContext netContext);

	void launchTestCompute() throws Exception;

	String getLineChartTitle();

	double getInitWeightRange(int idx);

	boolean isDropOutActive();


}