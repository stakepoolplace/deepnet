package RN;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;

import javax.xml.parsers.FactoryConfigurationError;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.log4j.xml.DOMConfigurator;

import RN.algoactivations.EActivation;
import RN.algotrainings.BackPropagationTrainer;
import RN.algotrainings.ITrainer;
import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.linkage.ELinkage;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import javafx.collections.FXCollections;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;

/**
 * @author Eric Marchand
 * 
 */
public class TestNetwork implements ITester {

	private static Logger logger = Logger.getLogger(TestNetwork.class);

	private static TestNetwork instance = null;
	private static String filepath = null;
	private static Integer timeSeriesOffset = 0;
	private static Integer inputSize = null;
	private static Integer layerHidden0NodesCount = null;
	private static Integer layerHidden1NodesCount = null;
	private static Integer layerHidden2NodesCount = null;
	private static Integer outputSize = null;
	private static boolean dropOutActive = false;

	private static String key = null;
	private static String value = null;
	private static Integer layerOutNodesCount = null;
	private static String layerOutActivation = null;
	private static Boolean layerOutRecurrent = null;
	private static String layerHidden0Activation = null;
	private static Boolean layerHidden0Recurrent = null;
	private static String layerHidden1Activation = null;
	private static Boolean layerHidden1Recurrent = null;
	private static String layerHidden2Activation = null;
	private static Boolean layerHidden2Recurrent = null;
	private static Integer layerInNodesCount = null;
	private static String layerInActivation = null;
	private static Boolean layerOutNodesDropOut = Boolean.FALSE;
	private static Boolean layerHidden0NodesDropOut = Boolean.FALSE;
	private static Boolean layerHidden1NodesDropOut = Boolean.FALSE;
	private static Boolean layerHidden2NodesDropOut = Boolean.FALSE;
	private static Boolean layerInNodesDropOut = Boolean.FALSE;
	private static Boolean lateralLinkRecurrentNodes = Boolean.FALSE;
	private static Boolean hiddensNodeNumberOptimized = Boolean.FALSE;
	private static int optimizedNumHiddens = 0;

	private static Integer trainingVectorNumber = null;

	public static INetwork network;

	public static List<LineChart.Series<Number, Number>> seriesInList = new ArrayList<LineChart.Series<Number, Number>>();
	public static List<LineChart.Series<Number, Number>> seriesOutList = new ArrayList<LineChart.Series<Number, Number>>();
	public static List<LineChart.Series<Number, Number>> seriesIdealList = new ArrayList<LineChart.Series<Number, Number>>();
	public final double[] initWeightRange = new double[] { 0D, 1.0D };
	public static Properties properties = null;

	static {

		properties = init();

		String timeSerieOffsetS = properties.getProperty("data.series.timeseries.offset");
		if (timeSerieOffsetS != null && !"".equals(timeSerieOffsetS))
			timeSeriesOffset = Integer.valueOf(timeSerieOffsetS);

		String fileS = properties.getProperty("data.series.filepath");
		if (fileS != null && !"".equals(fileS))
			filepath = fileS;

		String dropOutActiveStr = properties.getProperty("nn.dropout.active");
		if (dropOutActiveStr != null && !"".equals(dropOutActiveStr))
			dropOutActive = dropOutActiveStr.equalsIgnoreCase("true");

	}

	private TestNetwork() {
	}

	public static TestNetwork getInstance() {
		if (instance == null) {
			instance = new TestNetwork();
		}

		return instance;
	}

	public static void main(String[] args) throws Exception {
		ITester tester = TestNetwork.getInstance();
		ITrainer trainer = new BackPropagationTrainer();

		ViewerFX.startViewerFX();
		ViewerFX.setTrainer(trainer);
		ViewerFX.setTester(tester);

		int idx = 1;

		for (String sheetName : InputSample.getSheetsName(filepath)) {
			ViewerFX.excelSheets.add(new InputSample("Sheet" + idx + ": " + sheetName, ESamples.FILE, idx));
			idx++;
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.ITester#createNetwork(java.lang.Long, java.lang.Long)
	 */
	@Override
	public INetwork createNetwork(String networkName) {

		key = null;
		value = null;
		layerOutNodesCount = null;
		layerOutActivation = null;
		layerOutRecurrent = null;
		layerHidden0Activation = null;
		layerHidden0Recurrent = null;
		layerHidden1Activation = null;
		layerHidden1Recurrent = null;
		layerHidden2Activation = null;
		layerHidden2Recurrent = null;
		layerInNodesCount = null;
		layerInActivation = null;
		layerOutNodesDropOut = Boolean.FALSE;
		layerHidden0NodesDropOut = Boolean.FALSE;
		layerHidden1NodesDropOut = Boolean.FALSE;
		layerHidden2NodesDropOut = Boolean.FALSE;
		layerInNodesDropOut = Boolean.FALSE;
		lateralLinkRecurrentNodes = Boolean.FALSE;
		hiddensNodeNumberOptimized = Boolean.FALSE;

		for (final Object obj : properties.keySet()) {
			key = (String) obj;
			value = properties.getProperty(key);

			if (key.startsWith("nn.layer.out.nodes.count"))
				layerOutNodesCount = Integer.valueOf(value);

			if (key.startsWith("nn.layer.out.activation.fx"))
				layerOutActivation = value;
			if (key.startsWith("nn.layer.out.recurrent"))
				if (!"".equals(value))
					layerOutRecurrent = Boolean.valueOf(value);
			if (key.startsWith("nn.layer.out.nodes.dropout"))
				if (!"".equals(value))
					layerOutNodesDropOut = Boolean.valueOf(value);

			if (key.startsWith("nn.layer.hidden.0.nodes.count"))
				layerHidden0NodesCount = Integer.valueOf(value);
			if (key.startsWith("nn.layer.hidden.0.activation.fx"))
				layerHidden0Activation = value;
			if (key.startsWith("nn.layer.hidden.0.recurrent"))
				if (!"".equals(value))
					layerHidden0Recurrent = Boolean.valueOf(value);
			if (key.startsWith("nn.layer.hidden.0.nodes.dropout"))
				if (!"".equals(value))
					layerHidden0NodesDropOut = Boolean.valueOf(value);

			if (key.startsWith("nn.layer.hidden.1.nodes.count"))
				layerHidden1NodesCount = Integer.valueOf(value);
			if (key.startsWith("nn.layer.hidden.1.activation.fx"))
				layerHidden1Activation = value;
			if (key.startsWith("nn.layer.hidden.1.recurrent"))
				if (!"".equals(value))
					layerHidden1Recurrent = Boolean.valueOf(value);
			if (key.startsWith("nn.layer.hidden.1.nodes.dropout"))
				if (!"".equals(value))
					layerHidden1NodesDropOut = Boolean.valueOf(value);

			if (key.startsWith("nn.layer.hidden.2.nodes.count"))
				layerHidden2NodesCount = Integer.valueOf(value);
			if (key.startsWith("nn.layer.hidden.2.activation.fx"))
				layerHidden2Activation = value;
			if (key.startsWith("nn.layer.hidden.2.recurrent"))
				if (!"".equals(value))
					layerHidden2Recurrent = Boolean.valueOf(value);
			if (key.startsWith("nn.layer.hidden.2.nodes.dropout"))
				if (!"".equals(value))
					layerHidden2NodesDropOut = Boolean.valueOf(value);

			if (key.startsWith("nn.layer.in.nodes.count"))
				layerInNodesCount = Integer.valueOf(value);

			if (key.startsWith("nn.layer.in.activation.fx"))
				if (!"".equals(value))
					layerInActivation = value;
			if (key.startsWith("nn.layer.in.nodes.dropout"))
				if (!"".equals(value))
					layerInNodesDropOut = Boolean.valueOf(value);

			if (key.startsWith("nn.recurrent.nodes.linked"))
				if (!"".equals(value))
					lateralLinkRecurrentNodes = Boolean.valueOf(value);

			if (key.startsWith("nn.hidden.nodes.number.optimized"))
				if (!"".equals(value))
					hiddensNodeNumberOptimized = Boolean.valueOf(value);

		}

		INetwork network = Network.getInstance();
		network.setName(networkName);

		// Neuralware (2001) approach
		optimizedNumHiddens = getTrainingVectorNumber() / (5 * (getInputSize() + getOutputSize()));
		optimizedNumHiddens = Math.max(5, optimizedNumHiddens);

		Layer layer = new Layer(
				layerInActivation == null ? EActivation.IDENTITY : EActivation.getEnum(layerInActivation));
		Area area = new Area(inputSize != null ? inputSize : layerInNodesCount);
		layer.addArea(area);
		area.configureLinkage(ELinkage.ONE_TO_ONE, null, false);
		area.configureNode(false, EActivation.IDENTITY, ENodeType.REGULAR);
		area.createNodes(inputSize != null ? inputSize : layerInNodesCount);
		layer.setDropOut(layerInNodesDropOut);
		network.addLayer(layer);
		network.setTimeSeriesOffset(timeSeriesOffset);
		network.setRecurrentNodesLinked(lateralLinkRecurrentNodes);

		if (layerHidden0NodesCount != null && layerHidden0NodesCount > 0) {
			layer = new Layer(EActivation.getEnum(layerHidden0Activation));
			Area area0 = null;
			if (!hiddensNodeNumberOptimized) {
				area0 = new Area(layerHidden0NodesCount);
				layer.addArea(area0);
				area0.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
				area0.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area0.createNodes(layerHidden0NodesCount);
			} else {
				area0 = new Area(optimizedNumHiddens);
				layer.addArea(area0);
				area0.configureLinkage(ELinkage.MANY_TO_MANY, null, false);
				area0.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area0.createNodes(optimizedNumHiddens);
			}
			if (layerHidden0Recurrent != null)
				layer.setReccurent(layerHidden0Recurrent);
			layer.setDropOut(layerHidden0NodesDropOut);
			network.addLayer(layer);
		}

		if (layerHidden1NodesCount != null && layerHidden1NodesCount > 0) {
			layer = new Layer(EActivation.getEnum(layerHidden1Activation));
			Area area1 = null;
			if (!hiddensNodeNumberOptimized) {
				area1 = new Area(layerHidden1NodesCount);
				layer.addArea(area1);
				area1.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
				area1.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area1.createNodes(layerHidden1NodesCount);
			} else {
				area1 = new Area(optimizedNumHiddens);
				layer.addArea(area1);
				area1.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
				area1.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area1.createNodes(optimizedNumHiddens);
			}

			if (layerHidden1Recurrent != null)
				layer.setReccurent(layerHidden1Recurrent);
			layer.setDropOut(layerHidden1NodesDropOut);
			network.addLayer(layer);
		}

		if (layerHidden2NodesCount != null && layerHidden2NodesCount > 0) {
			layer = new Layer(EActivation.getEnum(layerHidden2Activation));
			Area area2 = null;
			if (!hiddensNodeNumberOptimized) {
				area2 = new Area(layerHidden2NodesCount);
				layer.addArea(area2);
				area2.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
				area2.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area2.createNodes(layerHidden2NodesCount);
			} else {
				area2 = new Area(optimizedNumHiddens);
				layer.addArea(area2);
				area2.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
				area2.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
				area2.createNodes(optimizedNumHiddens);
			}

			if (layerHidden2Recurrent != null)
				layer.setReccurent(layerHidden2Recurrent);
			layer.setDropOut(layerHidden2NodesDropOut);
			network.addLayer(layer);
		}

		layer = new Layer(EActivation.getEnum(layerOutActivation));
		Area area3 = new Area(outputSize != null ? outputSize : layerOutNodesCount);
		layer.addArea(area3);
		area3.configureLinkage(ELinkage.MANY_TO_MANY, null, true);
		area3.configureNode(true, EActivation.IDENTITY, ENodeType.REGULAR);
		area3.createNodes(outputSize != null ? outputSize : layerOutNodesCount);
		if (layerOutRecurrent != null)
			layer.setReccurent(layerOutRecurrent);
		layer.setDropOut(layerOutNodesDropOut);
		network.addLayer(layer);

		network.setName(network.getName() + network.geneticCodec());
		// network.show();

		TestNetwork.network = network;

		return network;

	}

	@Override
	public INetwork createXLSNetwork(String networkName, NetworkContext netContext) {

		TestNetwork.network = null;

		INetwork network = netContext.newNetwork(networkName);

		// network.show();

		TestNetwork.network = network;

		return network;

	}

	private static Properties init() throws FactoryConfigurationError {
		
		Properties properties;
		/**
		 * Mise en place log4j
		 */
		final InputStream log4jIs = TestNetwork.class.getResourceAsStream("/log4j.xml");
		new DOMConfigurator().doConfigure(log4jIs, LogManager.getLoggerRepository());

		/**
		 * Chargement du fichier de config
		 */
		final InputStream propertiesIs = TestNetwork.class.getResourceAsStream("/conf.properties");
		properties = new Properties();
		try {

			properties.load(propertiesIs);

		} catch (final IOException e) {

			final String message = "Unable to load properties file for Deepnet.";
			logger.error(message);

		} catch (final NullPointerException npe) {

			final String message = " property's missing.";
			logger.error(message);

		} finally {
			try {
				propertiesIs.close();
			} catch (IOException ignore) {
				// ignored
			}
			try {
				log4jIs.close();
			} catch (IOException ignore) {
				// ignored
			}
		}
		return properties;
	}

	public Double computeY(INode node, Double x) {
		List<Link> inputs = node.getInputs();
		return (-inputs.get(0).getWeight() * x + node.getBiasInput().getValue() * node.getBiasInput().getWeight())
				/ inputs.get(1).getWeight();
	}





	/*
	 * (non-Javadoc)
	 * 
	 * @see RN.ITester#launchRealCompute(java.lang.Long, java.lang.Long)
	 */
	@Override
	public void launchRealCompute() throws Exception {

		DataSeries dataSeries = DataSeries.getInstance();
		INetwork network = getNetwork();
		LineChart<Number, Number> lineChart = ViewerFX.lineChart;

		if (network != null && !dataSeries.getInputDataSet().isEmpty()) {

			OutputData output = null;

			if (lineChart.getData() == null)
				lineChart.setData(FXCollections.<XYChart.Series<Number, Number>>observableArrayList());

			LineChart.Series<Number, Number> series = null;
			initLineChartSeries();

			for (int idx = 0; idx < getInputsCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run in" + (lineChart.getData().size() + 1));
				seriesInList.add(series);
			}

			for (int idx = 0; idx < getOutputsCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run out[" + idx + "]" + (lineChart.getData().size() + 1));
				seriesOutList.add(series);
			}

			for (int idx = 0; idx < getOutputsCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run ideal[" + idx + "]" + (lineChart.getData().size() + 1));
				seriesIdealList.add(series);
			}

			int runCycle = 0;
			ListIterator<InputData> computeItr = dataSeries.getInputDataSet().listIterator();
			for (InputData entry : dataSeries.getInputDataSet()) {
				try {
					output = network.compute(computeItr);
				} catch (Exception e) {
					e.printStackTrace();
				}

				int idx = 0;
				if (ViewerFX.showLogs.isSelected()) {
					String log = "inputs[  ";
					for (double input : entry.getInput()) {
						log += input + "  ";
					}
					log += "]";
					log += "\r\noutputs[  ";
					for (double out : output.getOutput()) {
						log += out + "  ";
					}
					log += "]";
					log += "\r\nideals[  ";
					for (double ideal : entry.getIdeal()) {
						log += ideal + "  ";
					}

					log += "]\r\n";
					System.out.println(log);
					log = null;
				}

				idx = 0;
				for (LineChart.Series<Number, Number> seriesIn : seriesInList) {
					seriesIn.getData().add(new LineChart.Data<Number, Number>(runCycle, entry.getInput(idx++)));
				}
				idx = 0;
				for (LineChart.Series<Number, Number> seriesOut : seriesOutList) {
					seriesOut.getData().add(new LineChart.Data<Number, Number>(runCycle, output.getOutput(idx++)));
				}
				idx = 0;
				for (LineChart.Series<Number, Number> seriesIdeal : seriesIdealList) {
					seriesIdeal.getData().add(new LineChart.Data<Number, Number>(runCycle, entry.getIdeal(idx++)));
				}
				runCycle++;
			}
			ViewerFX.addSeriesToLineChart();
		}
	}

	@Override
	public void launchTestCompute() throws Exception {

		DataSeries dataSeries = DataSeries.getInstance();
		INetwork network = getNetwork();
		LineChart<Number, Number> lineChart = ViewerFX.lineChart;

		if (network != null && !dataSeries.testsAreEmpty()) {

			OutputData output = null;

			if (lineChart.getData() == null)
				lineChart.setData(FXCollections.<XYChart.Series<Number, Number>>observableArrayList());

			LineChart.Series<Number, Number> series = null;
			initLineChartSeries();

			for (int idx = 0; idx < getInputsCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run in" + (lineChart.getData().size() + 1));
				seriesInList.add(series);
			}

			for (int idx = 0; idx < getOutputsCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run out[" + idx + "]" + (lineChart.getData().size() + 1));
				seriesOutList.add(series);
			}

			int runCycle = 0;
			ListIterator<InputData> computeItr = dataSeries.getInputTestDataSet().listIterator();
			for (InputData entry : dataSeries.getInputTestDataSet()) {
				try {
					output = network.compute(computeItr);
				} catch (Exception e) {
					e.printStackTrace();
				}

				int idx = 0;
				if (ViewerFX.showLogs.isSelected()) {
					System.out.print("[  ");
					for (double input : entry.getInput()) {
						System.out.print(input + "  ");
					}
					System.out.print("]");

					for (double out : output.getOutput()) {
						System.out.print(", [actual=" + out);
					}
					System.out.println(" ");
				}

				idx = 0;
				for (LineChart.Series<Number, Number> seriesIn : seriesInList) {
					seriesIn.getData().add(new LineChart.Data<Number, Number>(runCycle, entry.getInput(idx++)));
				}
				idx = 0;
				for (LineChart.Series<Number, Number> seriesOut : seriesOutList) {
					seriesOut.getData().add(new LineChart.Data<Number, Number>(runCycle, output.getOutput(idx++)));
				}

				runCycle++;
			}
			ViewerFX.addSeriesToLineChart();
		}
	}

	private void initLineChartSeries() {
		seriesInList = new ArrayList<LineChart.Series<Number, Number>>();
		seriesOutList = new ArrayList<LineChart.Series<Number, Number>>();
		seriesIdealList = new ArrayList<LineChart.Series<Number, Number>>();
	}

	@Override
	public INetwork initWeights(double min, double max) {
		INetwork network = getNetwork();
		network.init(min, max);
		return network;
	}

	@Override
	public INetwork getNetwork() {
		return network;
	}

	@Override
	public int getInputsCount() {
		return network.getFirstLayer().getNodeCountMinusRecurrentOnes();
	}

	@Override
	public int getOutputsCount() {
		return network.getLastLayer().getNodeCount();
	}

	@Override
	public void setInputsCount(int insize) {
		this.inputSize = insize;
	}

	@Override
	public void setOutputsCount(int outsize) {
		this.outputSize = outsize;
	}

	public List<LineChart.Series<Number, Number>> getSeriesInList() {
		return seriesInList;
	}

	public void setSeriesInList(List<LineChart.Series<Number, Number>> seriesInList) {
		TestNetwork.seriesInList = seriesInList;
	}

	public List<LineChart.Series<Number, Number>> getSeriesOutList() {
		return seriesOutList;
	}

	public void setSeriesOutList(List<LineChart.Series<Number, Number>> seriesOutList) {
		TestNetwork.seriesOutList = seriesOutList;
	}

	public List<LineChart.Series<Number, Number>> getSeriesIdealList() {
		return seriesIdealList;
	}

	public void setSeriesIdealList(List<LineChart.Series<Number, Number>> seriesIdealList) {
		TestNetwork.seriesIdealList = seriesIdealList;
	}

	public void setNetwork(INetwork network) {
		TestNetwork.network = network;
	}

	@Override
	public double getInitWeightRange(int idx) {
		return initWeightRange[idx];
	}

	public String getFilePath() {
		return filepath;
	}

	public static void setFilepath(String filePath) {
		TestNetwork.filepath = filepath;
	}

	public Integer getTimeSeriesOffset() {
		return timeSeriesOffset;
	}

	public static void setTimeSeriesOffset(Integer timeSeriesOffset) {
		TestNetwork.timeSeriesOffset = timeSeriesOffset;
	}

	public static Integer getTrainingVectorNumber() {
		return trainingVectorNumber;
	}

	public void setTrainingVectorNumber(Integer trainingVectorNumber) {
		TestNetwork.trainingVectorNumber = trainingVectorNumber;
	}

	public static Integer getInputSize() {
		return inputSize;
	}

	public static Integer getOutputSize() {
		return outputSize;
	}

	@Override
	public String getLineChartTitle() {
		return "Please select a dataset";
	}

	public Integer getLayerHidden0NodesCount() {
		return layerHidden0NodesCount;
	}

	public void setLayerHidden0NodesCount(Integer layerHidden0NodesCount) {
		TestNetwork.layerHidden0NodesCount = layerHidden0NodesCount;
	}

	public Integer getLayerHidden1NodesCount() {
		return layerHidden1NodesCount;
	}

	public void setLayerHidden1NodesCount(Integer layerHidden1NodesCount) {
		TestNetwork.layerHidden1NodesCount = layerHidden1NodesCount;
	}

	public Integer getLayerHidden2NodesCount() {
		return layerHidden2NodesCount;
	}

	public void setLayerHidden2NodesCount(Integer layerHidden2NodesCount) {
		TestNetwork.layerHidden2NodesCount = layerHidden2NodesCount;
	}

	public Integer getLayerOutNodesCount() {
		return layerOutNodesCount;
	}

	public void setLayerOutNodesCount(Integer layerOutNodesCount) {
		TestNetwork.layerOutNodesCount = layerOutNodesCount;
	}

	public String getLayerOutActivation() {
		return layerOutActivation;
	}

	public void setLayerOutActivation(String layerOutActivation) {
		TestNetwork.layerOutActivation = layerOutActivation;
	}

	public Boolean getLayerOutRecurrent() {
		return layerOutRecurrent;
	}

	public void setLayerOutRecurrent(Boolean layerOutRecurrent) {
		TestNetwork.layerOutRecurrent = layerOutRecurrent;
	}

	public String getLayerHidden0Activation() {
		return layerHidden0Activation;
	}

	public void setLayerHidden0Activation(String layerHidden0Activation) {
		TestNetwork.layerHidden0Activation = layerHidden0Activation;
	}

	public Boolean getLayerHidden0Recurrent() {
		return layerHidden0Recurrent;
	}

	public void setLayerHidden0Recurrent(Boolean layerHidden0Recurrent) {
		TestNetwork.layerHidden0Recurrent = layerHidden0Recurrent;
	}

	public String getLayerHidden1Activation() {
		return layerHidden1Activation;
	}

	public void setLayerHidden1Activation(String layerHidden1Activation) {
		TestNetwork.layerHidden1Activation = layerHidden1Activation;
	}

	public Boolean getLayerHidden1Recurrent() {
		return layerHidden1Recurrent;
	}

	public void setLayerHidden1Recurrent(Boolean layerHidden1Recurrent) {
		TestNetwork.layerHidden1Recurrent = layerHidden1Recurrent;
	}

	public String getLayerHidden2Activation() {
		return layerHidden2Activation;
	}

	public void setLayerHidden2Activation(String layerHidden2Activation) {
		TestNetwork.layerHidden2Activation = layerHidden2Activation;
	}

	public Boolean getLayerHidden2Recurrent() {
		return layerHidden2Recurrent;
	}

	public void setLayerHidden2Recurrent(Boolean layerHidden2Recurrent) {
		TestNetwork.layerHidden2Recurrent = layerHidden2Recurrent;
	}

	public Integer getLayerInNodesCount() {
		return layerInNodesCount;
	}

	public void setLayerInNodesCount(Integer layerInNodesCount) {
		TestNetwork.layerInNodesCount = layerInNodesCount;
	}

	public String getLayerInActivation() {
		return layerInActivation;
	}

	public void setLayerInActivation(String layerInActivation) {
		TestNetwork.layerInActivation = layerInActivation;
	}

	public int getOptimizedNumHiddens() {
		return optimizedNumHiddens;
	}

	public void setOptimizedNumHiddens(int optimizedNumHiddens) {
		TestNetwork.optimizedNumHiddens = optimizedNumHiddens;
	}

	@Override
	public boolean isDropOutActive() {
		return dropOutActive;
	}

}
