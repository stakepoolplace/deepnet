package RN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.ListIterator;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import RN.algoactivations.ActivationFx;
import RN.algoactivations.EActivation;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.INode;
import javafx.animation.KeyFrame;
import javafx.animation.SequentialTransition;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.Event;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.geometry.VPos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.SceneAntialiasing;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ListView;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.Slider;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.RectangleBuilder;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;

public class ViewerFX extends Application {

	
	public static NumberAxis xAxis = null;
	public static NumberAxis yAxis = null;
	public static LineChart<Number, Number> lineChart = null;
	public static LineChart.Series<Number, Number> series = null;

	public static List<ActivationFx> activations = new ArrayList<ActivationFx>();
	final ObservableList<ActivationFx> layerFx = FXCollections.observableArrayList(activations);
	private static ESamples selectedSample = ESamples.COSINUS;
	protected EActivation selectedFxNode = EActivation.SYGMOID_0_1;

	protected static ITester tester = null;
	protected static ITrainer trainer = null;

	final Button train = new Button("Train");
	final Button trainStop = new Button("Stop train");
	final Button run = new Button("Run");
	final Button runTest = new Button("Run test");
	final Button remove = new Button("Remove last serie");
	final Button removeAll = new Button("Remove all series");
	final Button ldseries = new Button("Load Series");
	final Button load = new Button("Load network");
	final Button save = new Button("Save network");
	final Button rand = new Button("Randomize network");
	final Button clear = new Button("Clear data network");
	final Button print = new Button("Print network");

	public static CheckBox showSeriesIn = null;
//	public final static CheckBox showSeriesActFx = new CheckBox("Show activation(s) fx");
	public static CheckBox showSeriesOut = null;
	public static CheckBox showSeriesIdeals = null;
	public static CheckBox showLogs = null;
	public static CheckBox animatedTrain = null;


	public static List<ScheduledFuture> futures = new ArrayList<ScheduledFuture>();
	public static ScheduledFuture future = null;
	private static int threadPoolDelay = 500000;
	private static int THREAD_POOL_NUMBER = 5;
	private static ScheduledExecutorService scheduler = null;

	public static volatile boolean stopPool = false;
	public static volatile Integer factor = 0;

	private static SequentialTransition animation = new SequentialTransition();
	private static Timeline timeline1 = new Timeline();

	public static Slider coefActivationFx = null;
	public static ObservableList<InputSample> excelSheets = FXCollections.observableArrayList(new InputSample("No training set", ESamples.NONE));
	
	public static Rectangle clip = null;
	public static Pane hiddenGraphPane = new Pane();
	public static Pane hiddenGraphPane1 = new Pane();
	public static Pane hiddenGraphPane2 = new Pane();
	
	public static Pane scalingPane = new Pane();
	public static Pane graphPane = null;
	public static Pane netPane = null;

	public static boolean showLinearSeparation = false;

	

	static {
		activations.add(new ActivationFx("Cosinus", EActivation.COS));
		activations.add(new ActivationFx("Sinus", EActivation.SIN));
		activations.add(new ActivationFx("Identité", EActivation.IDENTITY));
		activations.add(new ActivationFx("Heaviside", EActivation.HEAVISIDE));
		activations.add(new ActivationFx("Sygmoide", EActivation.SYGMOID_0_1));
		activations.add(new ActivationFx("Tangente hyperbolique", EActivation.TANH));
	}

	@Override
	public void start(Stage stage) {

		stage.setTitle("Error rate");
		
		
		Group group = new Group();
		StackPane root = new StackPane(group);
		Scene scene = new Scene(root);
        stage.setScene(scene);
        
		
		createLineChart();
		graphPane = createGraph();
		netPane = createNetPane();
		
		
		TabPane tabPane = createTabPane();
		
		final StackPane stack = new StackPane();
		stack.getChildren().addAll(netPane, scalingPane, graphPane, lineChart);
		
		Pane commandsPane = createCommands();
		
		VBox vbox = new VBox();
		vbox.getChildren().addAll(tabPane, stack, commandsPane);
		
		
		root.getChildren().add(vbox);
		
		stage.show();


	}
	

	


	private TabPane createTabPane() {
		
		Tab tabNet = new Tab("Network");
		tabNet.setClosable(false);
		tabNet.selectedProperty().addListener(new ChangeListener<Boolean>(){

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				if(!oldValue && newValue){
					//Graphics3D.world.toFront();
					//lineChart.setVisible(true);
				}
				
			}
			
		});
		
		Tab tabError = new Tab("Training");
		tabError.setClosable(false);
		tabError.selectedProperty().addListener(new ChangeListener<Boolean>(){

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				if(!oldValue && newValue){
					lineChart.toFront();
					//lineChart.setVisible(true);
				}
				
			}
			
		});
		
		
		Tab tabGraph = new Tab("Graph by layer");
		tabGraph.setClosable(false);
		tabGraph.selectedProperty().addListener(new ChangeListener<Boolean>(){

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				if(!oldValue && newValue){
					graphPane.toFront();
					//graphPane.setVisible(true);
				}
				
			}
			
		});
		
		Tab tabScaling = new Tab("Scale network");
		tabScaling.setClosable(false);
		tabScaling.selectedProperty().addListener(new ChangeListener<Boolean>(){

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				if(!oldValue && newValue){
					scalingPane.toFront();
					//graphPane.setVisible(true);
				}
				
			}
			
		});
		
		TabPane tabPane = new TabPane(tabNet, tabError, tabGraph, tabScaling);
		tabPane.setManaged(true);
		tabPane.toFront();
		
		return tabPane;
	}

	private void createLineChart() {
		xAxis = new NumberAxis(0, 500, 10);
		yAxis = new NumberAxis(-0.1, 1, 0.1);
		lineChart = new LineChart<Number, Number>(xAxis, yAxis);
		lineChart.setStyle("-fx-background-color: #ffffff;");

		xAxis.setLabel("Training cycles");
		yAxis.setLabel("Error level");
		xAxis.setAnimated(false);
		yAxis.setAnimated(false);
		xAxis.setAutoRanging(true);
		yAxis.setAutoRanging(true);
		lineChart.setAnimated(false);

		lineChart.setTitle(tester.getLineChartTitle());
		lineChart.setPrefSize(1000, 400);
	}

	public Pane createCommands(){
		
		Pane commandPane = new Pane();
		
		showSeriesIn = new CheckBox("Show input(s)");
		showSeriesOut = new CheckBox("Show output(s)");
		showSeriesIdeals = new CheckBox("Show ideal(s)");
		showLogs = new CheckBox("Logs");
		animatedTrain = new CheckBox("animated");
		
		coefActivationFx = new Slider();
		

		final VBox vbox = new VBox();
		final HBox hbox0 = new HBox();
		final HBox hbox1 = new HBox();
		final HBox hbox2 = new HBox();
		vbox.getChildren().addAll(hbox0, hbox1, hbox2);
		commandPane.getChildren().add(vbox);
		

		final ComboBox<InputSample> comboSamples = new ComboBox<InputSample>(excelSheets);
		comboSamples.getSelectionModel().selectFirst();
		comboSamples.getSelectionModel().selectedItemProperty().addListener(new ChangeListener<InputSample>() {

			@Override
			public void changed(ObservableValue<? extends InputSample> arg0, InputSample arg1, InputSample arg2) {

				if (arg2 != null) {
					selectedSample = arg2.getSample();
					if (ESamples.FILE == selectedSample) {

						trainer.initTrainer();
						
						// clean 3D graphics world
						Graphics3D.clearShapes();

						try {
							InputSample.setFileSample(tester, tester.getFilePath(), arg2.getFileSheetIdx());
						} catch (Exception e1) {
							e1.printStackTrace();
						}

						lineChart.setTitle((tester instanceof TestNetwork ? "MLP " : "LSTM ") + "network : " + tester.getInputsCount() + " input(s), "
								+ tester.getOutputsCount() + " output(s)     DataSerie : " + DataSeries.getInstance().getInputDataSet().size() + " examples");

						System.out.println(DataSeries.getInstance().getString());
						
						// clean the linear separation graph
						hiddenGraphPane1.getChildren().clear();
						hiddenGraphPane2.getChildren().clear();
						
						scalingPane = createScaling();
						
						
						if (lineChart.getData() != null && !lineChart.getData().isEmpty())
							lineChart.setData(null);

						trainer.setInputDataSetIterator(null);
						if (series != null && series.getData() != null)
							series.getData().clear();

						try {
							tester.launchRealCompute();
						} catch (Exception e) {
							e.printStackTrace();
						}
						
						// Weights are randomized here because first propagation create weights in the 'unlinked' mode
						tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
						System.out.println("weigts randomized [ " + tester.getInitWeightRange(0) + ", " + tester.getInitWeightRange(1) + "] !");
						
					}

				}
			}
		});




		final TextField learningRateField = new TextField(String.valueOf(trainer.getLearningRate()));
		learningRateField.setPrefColumnCount(4);
		final TextField momentumField = new TextField(String.valueOf(trainer.getAlphaDeltaWeight()));
		momentumField.setPrefColumnCount(4);
		// final TextField momentumField = new
		// TextField(String.valueOf(TestNetwork.getAlphaDeltaWeight()));
		final Button submit = new Button("Change it!");

		showLogs.setSelected(true);

		showSeriesIn.setSelected(false);
		showSeriesIn.selectedProperty().addListener(new ChangeListener<Boolean>() {
			public void changed(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {
				if (new_val)
					lineChart.getData().addAll(tester.getSeriesInList());
				else
					lineChart.getData().removeAll(tester.getSeriesInList());
			}
		});



		showSeriesOut.setSelected(false);
		showSeriesOut.selectedProperty().addListener(new ChangeListener<Boolean>() {
			public void changed(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {
				if (new_val)
					lineChart.getData().addAll(tester.getSeriesOutList());
				else
					lineChart.getData().removeAll(tester.getSeriesOutList());
			}
		});

		showSeriesIdeals.setSelected(false);
		showSeriesIdeals.selectedProperty().addListener(new ChangeListener<Boolean>() {
			public void changed(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {
				if (new_val)
					lineChart.getData().addAll(tester.getSeriesIdealList());
				else
					lineChart.getData().removeAll(tester.getSeriesIdealList());
			}
		});






		Slider trainSpeedSlider = new Slider();
		trainSpeedSlider.setMin(9000);
		trainSpeedSlider.setMax(1000000);
		trainSpeedSlider.setValue(getThreadPoolDelay());
		trainSpeedSlider.setShowTickLabels(true);
		trainSpeedSlider.setShowTickMarks(true);
		trainSpeedSlider.setMajorTickUnit(500000);
		trainSpeedSlider.setMinorTickCount(10);
		trainSpeedSlider.setBlockIncrement(2);

		trainSpeedSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {

				if (old_val == null || new_val == null)
					return;

				if (future != null) {
					future.cancel(true);
					futures.remove(future);
					future = null;
				}
				setThreadPoolDelay(getThreadPoolDelay() + (new_val.intValue() - old_val.intValue()));
				future = scheduler.scheduleWithFixedDelay(task, 0, getThreadPoolDelay(), TimeUnit.MICROSECONDS);
				futures.add(future);

			}
		});

		Slider trainingSetSlider = new Slider();
		trainingSetSlider.setMin(0);
		trainingSetSlider.setMax(50000);
		trainingSetSlider.setValue(0);
		trainingSetSlider.setShowTickLabels(true);
		trainingSetSlider.setShowTickMarks(true);
		trainingSetSlider.setMajorTickUnit(25000);
		trainingSetSlider.setMinorTickCount(10);
		trainingSetSlider.setBlockIncrement(500);
		trainingSetSlider.setPrefWidth(1200);

		trainingSetSlider.valueProperty().addListener(new ChangeListener<Number>() {
			DataSeries data = DataSeries.getInstance();

			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {

				if (old_val == null || new_val == null)
					return;

				int counter = 0;

				try {
					while (counter++ < data.getInputDataSet().size()) {
						try {
							lineChart.setData(null);

							trainer.train();
							// absoluteError += trainer.getErrorRate();

						} catch (Exception e1) {
							System.out.println("Message" + e1.getMessage());
							e1.printStackTrace();
						}
					}
					tester.launchRealCompute();
					// System.out.println("Epoch" + counter);
				} catch (Throwable t) {
					System.out.println("Message" + t.getMessage());
				}

			}
		});



		animatedTrain.setSelected(false);
		animatedTrain.selectedProperty().addListener(new ChangeListener<Boolean>() {
			public void changed(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {

				if (new_val) {
					showLogs.setSelected(false);
				}
			}
		});


		Slider timeSeriesOffset = new Slider();
		timeSeriesOffset.setMin(0);
		timeSeriesOffset.setMax(300);
		timeSeriesOffset.setValue(tester.getTimeSeriesOffset());
		timeSeriesOffset.setShowTickLabels(true);
		timeSeriesOffset.setShowTickMarks(true);
		timeSeriesOffset.setMajorTickUnit(150);
		timeSeriesOffset.setMinorTickCount(5);
		timeSeriesOffset.setBlockIncrement(1.0D);

		timeSeriesOffset.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {

				if (old_val == null || new_val == null)
					return;

				INetwork network = tester.getNetwork();
				Integer offset = network.getTimeSeriesOffset() + (new_val.intValue() - old_val.intValue());
				tester.getNetwork().setTimeSeriesOffset(offset);
				lineChart.setData(null);
				try {
					tester.launchRealCompute();
				} catch (Exception e) {
					e.printStackTrace();
				}
				System.out.println("setting offset to : " + offset);
			}
		});

		coefActivationFx.setMin(0);
		coefActivationFx.setMax(5);
		coefActivationFx.setValue(1);
		coefActivationFx.setShowTickLabels(true);
		coefActivationFx.setShowTickMarks(true);
		coefActivationFx.setMajorTickUnit(1D);
		coefActivationFx.setMinorTickCount(5);
		coefActivationFx.setBlockIncrement(0.1D);

		coefActivationFx.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {

				if (old_val == null || new_val == null)
					return;
				System.out.println("setting coef to : " + new_val);
				lineChart.setData(null);
				try {
					tester.launchRealCompute();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});

		hbox0.setSpacing(10);
		hbox0.getChildren().addAll(trainingSetSlider);

		hbox1.setSpacing(10);
		hbox1.getChildren().addAll(animatedTrain, train, trainSpeedSlider, trainStop, run, runTest, remove, removeAll, rand, clear, print, learningRateField,
				momentumField, submit, showLogs);

		hbox2.setSpacing(10);
		hbox2.getChildren().addAll(comboSamples, coefActivationFx, showSeriesIn, showSeriesOut, showSeriesIdeals,
				timeSeriesOffset);


		
		hbox0.setPadding(new Insets(5, 0, 5, 20));
		hbox1.setPadding(new Insets(5, 20, 2, 20));
		hbox2.setPadding(new Insets(2, 20, 5, 20));
		
		
		train.setOnAction(new EventHandler<ActionEvent>() {

			@Override
			public void handle(ActionEvent e) {

				if (animatedTrain.isSelected()) {
					train.setText("Train x" + ++factor);
					startScheduledExecutorService();
				} else {
					try {

						trainer.getErrorLevelLines().clear();

						if (lineChart.getData() == null)
							lineChart.setData(FXCollections.<XYChart.Series<Number, Number>> observableArrayList());
						if (series == null) {
							series = new LineChart.Series<Number, Number>();
							lineChart.getData().add(series);
						}

						if (!lineChart.getData().contains(series))
							lineChart.getData().add(series);

						series.setName("Train " + (lineChart.getData().size() + 1));

						trainer.launchTrain(showLogs.isSelected());
						timeline1.setCycleCount(trainer.getErrorLevelLines().size());
						animation.play();

					} catch (Exception e1) {
						e1.printStackTrace();
					}
				}
			}
		});

		// create animation
		timeline1.getKeyFrames().add(

		new KeyFrame(Duration.millis(10), new EventHandler<ActionEvent>() {

			ListIterator<LineChart.Data<Number, Number>> errorItr = null;

			@Override
			public void handle(ActionEvent actionEvent) {

				if (!trainer.getErrorLevelLines().isEmpty()) {
					int nextIndex = 0;

					if (errorItr == null)
						errorItr = trainer.getErrorLevelLines().listIterator();
					else
						nextIndex = errorItr.nextIndex();

					try {
						if (errorItr.hasNext())
							series.getData().add(errorItr.next());
						else {
							errorItr = trainer.getErrorLevelLines().listIterator();
							if (errorItr.hasNext())
								series.getData().add(errorItr.next());
						}
					} catch (ConcurrentModificationException cme) {
						System.out.println("Concurrent modif KeyFrame :  nextIndex :" + nextIndex);
						//errorItr = trainer.getLines().listIterator(nextIndex);
					}
				}
			}

		})

		);

		animation.getChildren().add(timeline1);

		trainStop.setOnAction(new EventHandler<ActionEvent>() {

			@Override
			public void handle(ActionEvent e) {

				try {
					trainer.setBreakTraining(true);
					System.out.println("futures : " + futures.size());
					for (ScheduledFuture future : futures) {
						future.cancel(true);
					}
					train.setText("Train");
					factor = 0;
					Thread.sleep(500);
					java.util.ListIterator<ScheduledFuture> futuresItr = futures.listIterator();
					while (futuresItr.hasNext()) {
						ScheduledFuture future = futuresItr.next();
						if (future.isDone()) {
							futuresItr.remove();
							futures.remove(future);
						}
					}

				} catch (Exception e1) {
					e1.printStackTrace();
				}
			}
		});

		run.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				try {
					tester.launchRealCompute();
				} catch (Exception e1) {
					e1.printStackTrace();
				}
			}
		});

		runTest.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				try {
					tester.launchTestCompute();
				} catch (Exception e1) {
					e1.printStackTrace();
				}
			}
		});		
		
		remove.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				if (!lineChart.getData().isEmpty())
					lineChart.getData().remove(lineChart.getData().size() - 1);
			}
		});

		removeAll.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				if (!lineChart.getData().isEmpty())
					lineChart.setData(null);
				
//				createImageData();
				//loadImageData("file:C:\\Users\\Eric\\Pictures\\blank.png");
				//drawImageData();
			}
		});

		// Setting an action for the Submit button
		submit.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				if ((learningRateField.getText() != null && !learningRateField.getText().isEmpty())) {

					trainer.setLearningRate(Double.valueOf(learningRateField.getText()));
					trainer.setAlphaDeltaWeight(Double.valueOf(momentumField.getText()));
					System.out.println("learningRateField set to " + learningRateField.getText());
					System.out.println("momentumField set to " + momentumField.getText());
				} else {
					System.out.println("nothing happened !");
				}
			}
		});

		load.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				// NetworkService.loadNetwork(Long.valueOf(adapter.getText()),
				// Long.valueOf(network.getText()));
				System.out.println("nothing happened !");
			}
		});

		save.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				// NetworkService.saveNetwork(Long.valueOf(adapter.getText()),
				// Long.valueOf(network.getText()));
				System.out.println("nothing happened !");
			}
		});

		rand.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
				System.out.println("weigts randomized [ " + tester.getInitWeightRange(0) + ", " + tester.getInitWeightRange(1) + "] !");
			}
		});

		clear.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				DataSeries.getInstance().clearSeries();
				System.out.println("series input and ideals cleared !");
				trainer.getErrorLevelLines().clear();
				System.out.println("error level cleared !");
			}
		});

		print.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				System.out.println(tester.getNetwork().getString());
//				System.out.println(DataSeries.getInstance().getString());
			}
		});

		ldseries.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {

				// new Thread(new Runnable(){
				// @Override
				// public void run() {
				// RerateServices.insertOrder(Long.valueOf(adapter.getText()),
				// Long.valueOf(network.getText()));
				// AdminServices.jobRunRerating("2013_03", "yes");
				// }}).start();
				System.out.println("nothing happened !");
			}
		});
		
		return commandPane;
		
	}
	
	public Pane createNetPane() {
		
		return new HBox();
	}
	
	public Pane createGraph() {
		
		HBox hbox = new HBox();

		hbox.setStyle("-fx-background-color: #ffffff;-fx-border-color: #2e8b57;-fx-border-width: 1px;");

		hiddenGraphPane.setStyle("-fx-background-color: #ffffff;-fx-border-color: #2e8b57;-fx-border-width: 1px;");
		hiddenGraphPane.setMaxSize(400, 400);
		Line lineX = new Line(0, 200, 400, 200);
		lineX.setFill(Color.DARKGRAY);
		Text tX = createText(380, 200, "e1", VPos.BOTTOM);
		Line lineY = new Line(200, 0, 200, 400);
		lineY.setFill(Color.DARKGRAY);
		Text tY = createText(215, 20, "e2", VPos.BOTTOM);

		hiddenGraphPane.getChildren().addAll(lineX, tX, lineY, tY);

		hiddenGraphPane1.setMaxSize(400, 400);
		hiddenGraphPane2.setMaxSize(400, 400);
		clip = new Rectangle(0, 0, 400, 400);

		hiddenGraphPane.setClip(RectangleBuilder.create().width(clip.getWidth()).height(clip.getHeight()).build());
		hiddenGraphPane1.setClip(RectangleBuilder.create().width(clip.getWidth()).height(clip.getHeight()).build());
		hiddenGraphPane2.setClip(RectangleBuilder.create().width(clip.getWidth()).height(clip.getHeight()).build());

		StackPane stack = new StackPane(clip, hiddenGraphPane, hiddenGraphPane1, hiddenGraphPane2);

		hbox.getChildren().addAll(stack);

		return hbox;
	}
	   
	public Pane createScaling() {
		
		final INetwork network = tester.getNetwork();

		scalingPane.setStyle("-fx-background-color: #ffffff;-fx-border-color: #ff0000;-fx-border-width: 1px;");
		scalingPane.getChildren().clear();
		VBox vbox = new VBox();

		Slider scaleNode = null;
		for (int idxLayerHidden = 1; idxLayerHidden <= network.getLayers().size() - 2; idxLayerHidden++) {
			scaleNode = new Slider(0D, 1000D, network.getLayer(idxLayerHidden).getNodeCount());
			scaleNode.setMinorTickCount(5);
			scaleNode.setMajorTickUnit(50);
			scaleNode.setSnapToTicks(false);
			scaleNode.setShowTickLabels(true);
			scaleNode.setShowTickMarks(true);
			scaleNode.setPrefWidth(scalingPane.getWidth() - 50D);

			final int currentIdxLayer = idxLayerHidden;
			scaleNode.valueProperty().addListener(new ChangeListener<Number>() {
				public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {

					if (old_val == null || new_val == null)
						return;

					System.out.println(old_val.intValue() + " " + new_val.intValue());

					if (new_val.intValue() == 0)
						return;

					if (old_val.intValue() < new_val.intValue()) {

						for (int idx = 1; idx <= new_val.intValue() - old_val.intValue(); idx++) {
							INode node = network.getLayer(currentIdxLayer).getLayerNodes()
									.get(network.getLayer(currentIdxLayer).getNodeCount() - 1);
							INode nodeCopy = node.deepCopy();
							nodeCopy.randomizeWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
							node.getArea().addNode(nodeCopy);
						}

					} else if (old_val.intValue() > new_val.intValue()) {

						for (int idx = 1; idx <= old_val.intValue() - new_val.intValue(); idx++) {
							INode node = network.getLayer(currentIdxLayer).getLayerNodes()
									.get(network.getLayer(currentIdxLayer).getNodeCount() - 1);
							node.disconnect();
							((IArea)node.getArea()).removeNode(node);
						}

					}

				}
			});
			vbox.getChildren().add(scaleNode);
		}

		scalingPane.getChildren().add(vbox);
		
		return scalingPane;
	}
	   

	private Text createText(int x, int y, String label, VPos vPos) {
		Text text = new Text(x, y, label);
		text.setFill(Color.DARKGRAY);
		text.setFont(Font.font(Font.getDefault().getFamily(), 16));
		// text.rotateProperty().set( -theta );
		text.textAlignmentProperty().setValue(TextAlignment.CENTER);
		text.setX(text.getX() - text.getBoundsInLocal().getWidth() / 2.0);
		text.textOriginProperty().set(vPos);
		// if( vPos == VPos.BOTTOM ) {
		// text.setY( text.getY() + 1 );
		// }
		// else {
		// text.setY( text.getY() - 1 );
		// }
		return text;
	}



	void createNode(Stage stage, Group group) throws IOException {

		Pane myPane = (Pane) FXMLLoader.load(getClass().getResource("/com/optimitel/midas/RN/fxml/network.fxml"));
		Scene myScene = new Scene(myPane);
		stage.setScene(myScene);

	}

	public static void viewError() {
		launch();
	}

	public static void startViewerFX() {
		new Thread(new Runnable() {

			@Override
			public void run() {
				ViewerFX.viewError();
			}
		}).start();
	}

	public static void main(String[] args) {
		launch(args);
	}

	public static ESamples getSelectedSample() {
		return selectedSample;
	}

	public static void setSelectedSample(ESamples selectedSample) {
		ViewerFX.selectedSample = selectedSample;
	}

	public static void setTester(ITester instance) {
		ViewerFX.tester = instance;

	}

	public static void setTrainer(ITrainer trainer) {
		ViewerFX.trainer = trainer;
	}

	// public volatile static int counter = 1;
	private void startScheduledExecutorService() {

		scheduler = Executors.newScheduledThreadPool(THREAD_POOL_NUMBER);
		future = scheduler.scheduleWithFixedDelay(task, 0, threadPoolDelay, TimeUnit.MICROSECONDS);
		futures.add(future);

	}

	private Runnable task = new Runnable() {

		volatile int counter = 1;
		volatile double absoluteError = 0.0D;

		DataSeries data = DataSeries.getInstance();

		@Override
		public void run() {

			counter++;
			// if (!stopPool) {
			// System.out.println("task1: " + Thread.currentThread().getName());

			if (counter <= data.getInputDataSet().size() * 50000) {
				// if (((float)counter / dataSerieSize) % 100 == 0)

				Platform.runLater(new Runnable() {
					@Override
					public void run() {
						int dataSetCounter = 0;
						try {

							while (dataSetCounter++ < data.getInputDataSet().size()) {
								try {
									lineChart.setData(null);

									trainer.train();

//									drawImageData();
									absoluteError += trainer.getErrorRate();
									
								} catch (Exception e1) {
									System.out.println("Message" + e1.getMessage());
									e1.printStackTrace();
								}
							}
							
							if(showLinearSeparation)
								tester.showLinearSeparation();
							
							tester.launchRealCompute();
							
						} catch (Throwable t) {
							System.out.println("Message" + t.getMessage());
						}
					}
				});

			} else {
				// scheduler.shutdown();
				//train.setText("Train");
				factor = 0;
				Platform.runLater(new Runnable() {
					@Override
					public void run() {
						System.out.println("fin du training. Nb valeurs:" + counter + "  erreur:" + absoluteError / 2.0D);
						scheduler.shutdown();
						counter = 1;
						
						try {
							trainer.setBreakTraining(true);
							System.out.println("futures : " + futures.size());
							for (ScheduledFuture future : futures) {
								future.cancel(true);
							}
							train.setText("Train");
							factor = 0;
							Thread.sleep(500);
							java.util.ListIterator<ScheduledFuture> futuresItr = futures.listIterator();
							while (futuresItr.hasNext()) {
								ScheduledFuture future = futuresItr.next();
								if (future.isDone()) {
									futuresItr.remove();
									futures.remove(future);
								}
							}

						} catch (Exception e1) {
							e1.printStackTrace();
						}
					}
				});
			}
		}
	};

	public static void addSeriesToLineChart() {
		if (showSeriesIn.isSelected())
			lineChart.getData().addAll(tester.getSeriesInList());
		if (showSeriesOut.isSelected())
			lineChart.getData().addAll(tester.getSeriesOutList());
		if (showSeriesIdeals.isSelected())
			lineChart.getData().addAll(tester.getSeriesIdealList());
//		if (showSeriesActFx.isSelected())
//			lineChart.getData().addAll(tester.getSeriesActFxList());
	}

	public static int getThreadPoolDelay() {
		return threadPoolDelay;
	}

	public static void setThreadPoolDelay(int threadPoolDelay) {
		ViewerFX.threadPoolDelay = threadPoolDelay;
	}
	
//	private void createImageData() {
//        int i = 0;
//        for (int y = 0; y < IMAGE_HEIGHT; y++) {
//            int r = y * 255 / IMAGE_HEIGHT;
//            for (int x = 0; x < IMAGE_WIDTH; x++) {
//                int g = x * 255 / IMAGE_WIDTH;
//                imageData[i] = (byte) 1;
//                imageData[i + 1] = (byte) 1;
//                imageData[i + 2] = (byte) 1;
//                i += 3;
//            }
//        }
//    }
//	
//	private void loadImageData(String path){
//		Image image = new Image(path);
//		PixelReader pixelReader = image.getPixelReader();
//        System.out.println("Image Width: " + image.getWidth());
//        System.out.println("Image Height: " + image.getHeight());
//        System.out.println("Pixel Format: " + pixelReader.getPixelFormat());
//        
//        int i = 0;
//        // Determine the color of each pixel in the image
//        for (int readY = 0; readY < image.getHeight(); readY++) {
//            for (int readX = 0; readX < image.getWidth(); readX++) {
//                Color color = pixelReader.getColor(readX, readY);
////                System.out.println("\nPixel color at coordinates ("
////                        + readX + "," + readY + ") "
////                        + color.toString());
////                System.out.println("R = " + color.getRed());
////                System.out.println("G = " + color.getGreen());
////                System.out.println("B = " + color.getBlue());
////                System.out.println("Opacity = " + color.getOpacity());
////                System.out.println("Saturation = " + color.getSaturation());
//                imageData[i] = (byte) (color.getRed() * 255);
//                imageData[i + 1] = (byte) (color.getGreen() * 255);
//                imageData[i + 2] = (byte) (color.getBlue() * 255);
//                i += 3;
//            }
//        }
//	}
// 
//	public static void createImageData(double input1, double input2, double red, double green, double blue) {
//        int x = (int) (input1 * IMAGE_WIDTH);
//        int y = (int) (input2 * IMAGE_HEIGHT);
//        int idx;
//        if(y == 0){
//        	idx = x * 3;
//        }else{
//        	idx =  (y * IMAGE_WIDTH - (IMAGE_WIDTH - x)) * 3 ;	
//        }
//        
//        if(idx < IMAGE_WIDTH * IMAGE_WIDTH * 3){
//        	imageData[idx] = (byte) (red * 255);
//        	imageData[idx + 1] = (byte) (green * 255);
//        	imageData[idx + 2] = (byte) (blue * 255);
//        }
//        
//    }
	
//	public static void createImageData(List<INode> input1) {
//        int idx = 0;
//		for (INode node : input1) {
//			for (Link link : node.getInputs()) {
//				if (idx < IMAGE_WIDTH * IMAGE_WIDTH * 3) {
//					imageData[idx] = (byte) (255);
//					imageData[idx + 1] = (byte) (link.getValue() * 255);
//					imageData[idx + 2] = (byte) (255);
//					idx = idx + 3;
//				}
//			}
//			//premier neurone seulement
//			if(node.getNodeId() == 0)
//			break;
//		}
//        
//    }
//	
//    public static void drawImageData(int width, int height) {
//    	
//        PixelWriter pixelWriter = gc.getPixelWriter();
//        
//        pixelWriter.setPixels(0, 0, width, height, pixelFormat, imageData, 0, width * 3);
//
//        // Add drop shadow effect
////        gc.applyEffect(new DropShadow(20, 20, 20, Color.GRAY));
//    }

}