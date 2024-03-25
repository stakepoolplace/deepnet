package RN;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import javafx.collections.FXCollections;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;

import javax.xml.parsers.FactoryConfigurationError;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.log4j.xml.DOMConfigurator;

import RN.algotrainings.ITrainer;
import RN.algotrainings.LSTMTrainer;
import RN.dataset.InputData;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import dmonner.xlbp.Network;
import dmonner.xlbp.UniformWeightInitializer;
import dmonner.xlbp.WeightUpdaterType;
import dmonner.xlbp.compound.InputCompound;
import dmonner.xlbp.compound.MemoryCellCompound;
import dmonner.xlbp.compound.XEntropyTargetCompound;
import dmonner.xlbp.trial.Step;

/**
 * @author Eric Marchand
 * 
 */
public class TestLSTMNetwork implements ITester {

	private static Logger logger = Logger.getLogger(TestLSTMNetwork.class);
	
	private static TestLSTMNetwork instance;

	public static INetwork net = null;

	InputCompound in = null;
	XEntropyTargetCompound out = null;

	final public static int trialLength = 500;
	final static int epochs = 10;
	static int insize = 4;
	final static int hidsize = 10;
	static int outsize = 2;
	// The number of trials in an epoch; i.e. size of group to measure. Default
	// = 100.
	final static int trialsPerEpoch = 4;

	Iterator<InputData> inputDataSetIterator = null;
	LineChart<Number, Number> sc = null;
	public static List<LineChart.Series<Number, Number>> seriesInList = new ArrayList<LineChart.Series<Number, Number>>();
	public static List<LineChart.Series<Number, Number>> seriesOutList = new ArrayList<LineChart.Series<Number, Number>>();
	public static List<LineChart.Series<Number, Number>> seriesIdealList = new ArrayList<LineChart.Series<Number, Number>>();	
	public static Properties properties = null;
	
	private static String filepath = null;
	private static Integer timeSeriesOffset = 0;
	
	static {

		properties = init();

		String timeSerieOffsetS = properties.getProperty("data.series.timeseries.offset");
		if (timeSerieOffsetS != null && !"".equals(timeSerieOffsetS))
			timeSeriesOffset = Integer.valueOf(timeSerieOffsetS);

		String fileS = properties.getProperty("data.series.filepath");
		if (fileS != null && !"".equals(fileS))
			filepath = fileS;
		
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

			final String message = "Unable to load properties file for Midas.";
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
	
	public TestLSTMNetwork() {
	}

	public static void main(String[] args) throws IOException {
		ITester tester = TestLSTMNetwork.getInstance();
		ITrainer trainer = LSTMTrainer.getInstance();
//		DataSeries.getInstance().setTrainer(trainer);
		
//		InputSample.setSamples(1L, 1L, insize, outsize, delta, delay);
		
//        InputSample.setFileSample("src/RN/Sinus.xls", 6, 1L, 1L);
		
		ViewerFX.startViewerFX();
		ViewerFX.setTester(tester);
		ViewerFX.setTrainer(trainer);

		int idx = 1;
		
		for (String sheetName : InputSample.getSheetsName(filepath)) {
			ViewerFX.excelSheets.add(new InputSample("Sheet" + idx + ": " + sheetName, ESamples.FILE, idx));
			idx++;
		}
		
		// instance.trainLSTMNetwork();
	}

	public static TestLSTMNetwork getInstance() throws IOException {
		if (instance == null) {
			instance = new TestLSTMNetwork();
		}

		return instance;
	}


	
	public Network createLSTMNetwork() throws IOException {

		final String mctype = "IFOP";

		in = new InputCompound("Input", insize);
		final MemoryCellCompound mc = new MemoryCellCompound("Hidden", hidsize, mctype);
		final MemoryCellCompound mc2 = new MemoryCellCompound("Hidden2", hidsize, mctype);
		out = new XEntropyTargetCompound("Output", outsize);

		out.addUpstreamWeights(mc2);
		mc2.addUpstreamWeights(mc);
		mc.addUpstreamWeights(in);

		NetLSTMNetwork net = new NetLSTMNetwork("CanonicalLSTM");
		net.setWeightUpdaterType(WeightUpdaterType.basic(0.1F));
		net.setWeightInitializer(new UniformWeightInitializer(1.0F, -0.1F, 0.1F));
		net.add(in);
		net.add(mc);
		net.add(mc2);
		net.add(out);
		net.optimize();
		net.build();
		
		TestLSTMNetwork.net = net;

		// System.out.println(net.toString("NAWL"));

		// FileWriter outfile = new FileWriter(new File("test.dot"));
		// outfile.write((new NetworkDotBuilder(net)).toString());
		// outfile.flush();
		// outfile.close();

		// -- Print the network
		// System.out.println(net.toString("NICXW"));
		System.out.println("Creation of network LSTM\t#inputs: " + insize + " #outputs:" + outsize );
		System.out.println("Total weights: " + net.nWeights() + "\n");

//		net.getExitPoint(0).getDownstream().getUpstream().
		
		return net;

	}

	@Override
	public INetwork createNetwork(String name) {
		INetwork net = null;
		try {
			createLSTMNetwork();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return net;
	}

	@Override
	public void launchRealCompute() {

		DataSeries dataSeries = DataSeries.getInstance();
		ITrainer trainer = LSTMTrainer.getInstance();

		sc = ViewerFX.lineChart;		
		if (sc.getData() == null){
			sc.setData(FXCollections.<XYChart.Series<Number, Number>> observableArrayList());
		}
		
		
		LineChart.Series<Number, Number> series = null;
		seriesInList = new ArrayList<LineChart.Series<Number, Number>>();
		seriesOutList = new ArrayList<LineChart.Series<Number, Number>>();
		seriesIdealList = new ArrayList<LineChart.Series<Number, Number>>();
		
		for(int idx = 0; idx < getInputsCount(); idx++){
			series = new LineChart.Series<Number, Number>();
			series.setName("Run in" + (sc.getData().size() + 1));
			seriesInList.add(series);
		}
		
	
		for(int idx = 0; idx < getOutputsCount(); idx++){
			series = new LineChart.Series<Number, Number>();
			series.setName("Run out" + (sc.getData().size() + 1));
			seriesOutList.add(series);
		}
		

		for(int idx = 0; idx < getOutputsCount(); idx++){
			series = new LineChart.Series<Number, Number>();
			series.setName("Run ideal" + (sc.getData().size() + 1));
			seriesIdealList.add(series);
		}
		
		int runCycle = 0;
		for (InputData entry : dataSeries.getInputDataSet()) {

			// Clears the network, setting all units to their default activation
			// levels
			// net.clear();

			// Impose an input vector on the units of the input layer
			in.setInput(entry.getInputArray());

			((NetLSTMNetwork) net).activateTest();

			int idx = 0;
			for(LineChart.Series<Number, Number> seriesIn : seriesInList){
				seriesIn.getData().add(new ScatterChart.Data<Number, Number>(runCycle, in.getOutput().getActivations()[idx++]));
			}
			idx = 0;
			for(LineChart.Series<Number, Number> seriesOut : seriesOutList){
				seriesOut.getData().add(new ScatterChart.Data<Number, Number>(runCycle, out.getOutput().getActivations()[idx++]));
			}
			idx = 0;
			for(LineChart.Series<Number, Number> seriesIdeal : seriesIdealList){
				seriesIdeal.getData().add(new ScatterChart.Data<Number, Number>(runCycle, entry.getIdeal(idx++)));
			}
			runCycle++;
		}
		



		

	}

	// @Override
	// public Trial nextTestTrial()
	// {
	// return nextTrainTrial();
	// }
	//
	// @Override
	// public Trial nextTrainTrial()
	// {
	// final Trial trial = new Trial(getMetaNetwork());
	//
	// for(int i = 0; i < trialLength; i++)
	// {
	// // Adds a new Step to your Trial; the Trial automatically keeps track of
	// the steps & order.
	// final Step step = trial.nextStep();
	// // Optional: Set the sub-Network to activate on this step; defaults to
	// Trial's Network set above.
	// // step.setNetwork(stepNet);
	// // Note: You can do either or both of the following on a single Step; if
	// you have more than
	// // one simultaneous input or target (on different input/output layers) on
	// the same step, use the
	// // variants of these methods that allow you to specify the Layer the
	// input or target applies to.
	// // step.addInput(inputPattern);
	// // step.addTarget(targetPattern);
	//
	// setSamples(step, insize, outsize, delta, delay);
	//
	// // Optional: Tell the Trial to record the activation of the given Layer
	// after this step; these
	// // recordings can be saved and used for post-hoc analysis.
	// // step.addRecord(layerToRecord);
	// }
	//
	// return trial;
	// }
	//
	// @Override
	// public Trial nextValidationTrial()
	// {
	//
	// return nextTrainTrial();
	//
	// }
	//
	//
	//
	// @Override
	// public int nTestTrials()
	// {
	// return 1;
	// }
	//
	// @Override
	// public int nTrainTrials()
	// {
	// return trialsPerEpoch;
	// }
	//
	// @Override
	// public int nValidationTrials()
	// {
	// return 1;
	// }
	//


	static void setSamples(Step step, int inputCount, int outputCount, double delta, double delay) {
		InputSample.initIkeda();
		double d1 = (inputCount + outputCount - 2) * delta + delay;
		for (int j = 0; j < 500; j++) {
			float[] inputList = new float[inputCount];
			float[] outputList = new float[outputCount];
			double d2 = j * (1.0D - d1) / 500.0D;
			double d3;
			for (int i = 0; i < inputCount; i++) {
				d3 = d2 + delta * i;
				// inputList.add(InputSample.compute(ESamples.COSINUS, d3));
//				inputList[i] = (float) InputSample.getInstance().compute(ESamples.CHAOS, d3);
			}
			for (int i = 0; i < outputCount; i++) {
				d3 = d2 + delay + delta * (i + inputCount - 1);
				// outputList.add(InputSample.compute(ViewerFX.getSelectedSample(),
				// d3));
//				outputList[i] = (float) InputSample.getInstance().compute(ESamples.CHAOS, d3);
			}
			step.addInput(inputList);
			step.addTarget(outputList);
			// add(1L, 1L, inputList, outputList);
		}
	}


	@Override
	public int getInputsCount() {
		return insize;
	}

	@Override
	public int getOutputsCount() {
		return outsize;
	}

	@Override
	public void setInputsCount(int insize) {
		TestLSTMNetwork.insize = insize;
	}

	@Override
	public void setOutputsCount(int outsize) {
		TestLSTMNetwork.outsize = outsize;
	}

	public static INetwork getNet() {
		return net;
	}

	public InputCompound getIn() {
		return in;
	}

	public XEntropyTargetCompound getOut() {
		return out;
	}


	@Override
	public INetwork initWeights(double min, double max) {
		((NetLSTMNetwork) net).clear();
		return null;
	}

	


	@Override
	public INetwork getNetwork() {
		return net;
	}

	@Override
	public List<LineChart.Series<Number, Number>> getSeriesInList() {
		return seriesInList;
	}

	@Override
	public void setSeriesInList(List<LineChart.Series<Number, Number>> seriesInList) {
		TestLSTMNetwork.seriesInList = seriesInList;
	}

	@Override
	public List<LineChart.Series<Number, Number>> getSeriesOutList() {
		return seriesOutList;
	}

	@Override
	public void setSeriesOutList(List<LineChart.Series<Number, Number>> seriesOutList) {
		TestLSTMNetwork.seriesOutList = seriesOutList;
	}

	@Override
	public List<LineChart.Series<Number, Number>> getSeriesIdealList() {
		return seriesIdealList;
	}

	@Override
	public void setSeriesIdealList(List<LineChart.Series<Number, Number>> seriesIdealList) {
		TestLSTMNetwork.seriesIdealList = seriesIdealList;
	}



	@Override
	public void setNetwork(INetwork net) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getInitWeightRange(int idx) {
		// TODO Auto-generated method stub
		return 0D;
	}
	
	

	@Override
	public String getLineChartTitle() {
		return "LSTM network : " + insize + " input(s), " + outsize + " output(s)";
	}
	
	@Override
	public String getFilePath() {
		return filepath;
	}

	@Override
	public Integer getTimeSeriesOffset() {
		return 0;
	}

	@Override
	public void setTrainingVectorNumber(Integer lastRowNum) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getLayerHidden0NodesCount() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden0NodesCount(Integer layerHidden0NodesCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getLayerHidden1NodesCount() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden1NodesCount(Integer layerHidden1NodesCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getLayerHidden2NodesCount() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden2NodesCount(Integer layerHidden2NodesCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getLayerOutNodesCount() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerOutNodesCount(Integer layerOutNodesCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getLayerOutActivation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerOutActivation(String layerOutActivation) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Boolean getLayerOutRecurrent() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerOutRecurrent(Boolean layerOutRecurrent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getLayerHidden0Activation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden0Activation(String layerHidden0Activation) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Boolean getLayerHidden0Recurrent() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden0Recurrent(Boolean layerHidden0Recurrent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getLayerHidden1Activation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden1Activation(String layerHidden1Activation) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Boolean getLayerHidden1Recurrent() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden1Recurrent(Boolean layerHidden1Recurrent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getLayerHidden2Activation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden2Activation(String layerHidden2Activation) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Boolean getLayerHidden2Recurrent() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerHidden2Recurrent(Boolean layerHidden2Recurrent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Integer getLayerInNodesCount() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerInNodesCount(Integer layerInNodesCount) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getLayerInActivation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setLayerInActivation(String layerInActivation) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getOptimizedNumHiddens() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setOptimizedNumHiddens(int optimizedNumHiddens) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public INetwork createXLSNetwork(String networkName, NetworkContext netContext) {


		INetwork network = netContext.newNetwork(networkName);

		// network.show();

		TestLSTMNetwork.net = network;

		return network;
	}

	@Override
	public void launchTestCompute() throws Exception {
		// TODO Auto-generated method stub
		
	}

	public static String getFilepath() {
		return filepath;
	}

	public static void setFilepath(String filepath) {
		TestLSTMNetwork.filepath = filepath;
	}

	public static void setTimeSeriesOffset(Integer timeSeriesOffset) {
		TestLSTMNetwork.timeSeriesOffset = timeSeriesOffset;
	}

	@Override
	public boolean isDropOutActive() {
		return false;
	}



}
