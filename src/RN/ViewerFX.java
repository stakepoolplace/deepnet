package RN;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

import RN.algoactivations.ActivationFx;
import RN.algoactivations.EActivation;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
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
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.shape.Rectangle;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Duration;

/**
 * @author Eric Marchand
 * 
 */
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
	final Button save = new Button("Save network");
	final Button load = new Button("Load network");
	final Button rand = new Button("Randomize network");
	final Button clear = new Button("Clear");
	final Button print = new Button("Print network");
	final Button printDeserializedNet = new Button("Print deserialized network");
	
	static TabPane tabPane = null;
	static Tab tabConsole = null;
	static Tab tabTraining = null;
	static Tab tabScaling = null;
	
	public final static int nbCycles = 500;


	public static CheckBox showLogs = null;

	private static int threadPoolDelay = 500000;

	public static volatile boolean stopPool = false;
	public static volatile Integer factor = 0;

	private static SequentialTransition animation = new SequentialTransition();
	private static Timeline timeline1 = new Timeline();

	public static ObservableList<InputSample> excelSheets = FXCollections
			.observableArrayList(new InputSample("Please select a dataset", ESamples.NONE));

	public static Rectangle clip = null;

	public static Pane scalingPane = new Pane();
	public static Pane netPane = null;
	private static TextArea consoleTextArea; // Référence globale pour l'area de texte de la console

	public static boolean showLinearSeparation = false;

	// Ajoutez un champ pour stocker la dernière valeur du cycle d'entraînement
	private static int lastTrainingCycle = 0;
	private static int origTrainingCycle = 0;
	
	
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

	    createLineChart(); // Initialiser le lineChart
	    netPane = createNetPane();
	    scalingPane = createScaling(); // Initialiser le scalingPane
	    createConsoleTab(); // Initialiser le consoleTextArea

	    tabPane = createTabPane(); // Cette méthode va maintenant configurer correctement les onglets

	    Pane commandsPane = createCommands();

	    VBox vbox = new VBox(tabPane, commandsPane); // Utiliser tabPane directement sans le stack non nécessaire

	    root.getChildren().add(vbox);

	    // Configuration initiale des boutons
	    configureInitialButtonState();

	    stage.show();

	}
	
	private void configureInitialButtonState() {
	    train.setDisable(true);
	    trainStop.setDisable(true);
	    run.setDisable(true);
	    runTest.setDisable(true);
	    save.setDisable(true);
	    rand.setDisable(true);
	    clear.setDisable(true);
	    print.setDisable(true);
	    printDeserializedNet.setDisable(true);
	}

	
	private static void createConsoleTab() {
	    consoleTextArea = new TextArea(); // Initialiser l'area de texte de la console
	    consoleTextArea.setEditable(false); // Rendre non éditable
	    consoleTextArea.textProperty().addListener((observable, oldValue, newValue) -> {
			Platform.runLater(() -> {
		    	consoleTextArea.setScrollTop(Double.MAX_VALUE); // Défiler automatiquement vers le bas
			});
	    });

	    
	}

	private TabPane createTabPane() {
		
	    TabPane tabPane = new TabPane();

	    tabTraining = new Tab("Training", lineChart); // Ajouter le lineChart directement à l'onglet
	    tabTraining.setClosable(false);

	    tabScaling = new Tab("Scale network", scalingPane); // Ajouter le scalingPane directement à l'onglet
	    tabScaling.setClosable(false);

	    tabConsole = new Tab("Console", consoleTextArea); // Ajouter le consoleTextArea directement à l'onglet
	    tabConsole.setClosable(false);

	    tabPane.getTabs().addAll(tabTraining, tabScaling, tabConsole); // Ajouter tous les onglets au TabPane

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

	public Pane createCommands() {

		Pane commandPane = new Pane();

		showLogs = new CheckBox("Logs");

		final VBox vbox = new VBox();
		final HBox hbox1 = new HBox();
		final HBox hbox2 = new HBox();
		vbox.getChildren().addAll(hbox1, hbox2);
		commandPane.getChildren().add(vbox);

		final ComboBox<InputSample> comboSamples = new ComboBox<InputSample>(excelSheets);
		comboSamples.getSelectionModel().selectFirst();
		comboSamples.getSelectionModel().selectedItemProperty().addListener(new ChangeListener<InputSample>() {

			@Override
			public void changed(ObservableValue<? extends InputSample> arg0, InputSample arg1, InputSample arg2) {

				if (arg2 != null) {
					selectedSample = arg2.getSample();

					// Activer le bouton "Train" si un dataset valide est sélectionné, sinon le
					// désactiver
					train.setDisable(selectedSample == ESamples.NONE);
					trainStop.setDisable(selectedSample == ESamples.NONE);
					run.setDisable(selectedSample == ESamples.NONE);
					runTest.setDisable(selectedSample == ESamples.NONE);
					save.setDisable(selectedSample == ESamples.NONE);
					rand.setDisable(selectedSample == ESamples.NONE);
					clear.setDisable(selectedSample == ESamples.NONE);
					print.setDisable(selectedSample == ESamples.NONE);
					printDeserializedNet.setDisable(selectedSample == ESamples.NONE);
					
		            lastTrainingCycle = 0; // Réinitialisez la dernière valeur du cycle d'entraînement
		            origTrainingCycle = 0;
		            if (!lineChart.getData().isEmpty())
		                lineChart.getData().clear(); // Nettoyez toutes les séries de données du graphique
		            series = null; // Réinitialisez la série
		            
	                xAxis.setAutoRanging(false); // Désactivez l'auto-range pour définir manuellement les limites
	                xAxis.setLowerBound(origTrainingCycle); // Démarrez à partir de la dernière valeur du cycle d'entraînement
	                xAxis.setUpperBound(lastTrainingCycle + nbCycles);

					if (ESamples.FILE == selectedSample) {

						trainer.initTrainer();

						// clean 3D graphics world
						Graphics3D.clearShapes();

						try {
							InputSample.setFileSample(tester, tester.getFilePath(), arg2.getFileSheetIdx());
						} catch (Exception e1) {
							e1.printStackTrace();
						}
						
						boolean isMLP = tester instanceof TestNetwork;

						if(isMLP) {
							lineChart.setTitle("MLP network : "
									+ tester.getNetwork().getLayers().size() + " layers, " 
									+ "  DataSerie : " + DataSeries.getInstance().getInputDataSet().size()
									+ " examples");		
						} else {
							lineChart.setTitle("LSTM network : "
									+ tester.getInputsCount() + " input(s), " + tester.getOutputsCount()
									+ " output(s)     DataSerie : " + DataSeries.getInstance().getInputDataSet().size()
									+ " examples");
						}

						Platform.runLater(() -> {
							tabPane.getSelectionModel().select(tabConsole);
						});
						consoleTextArea.setText(DataSeries.getInstance().getString()+ "\n");

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

						// Weights are randomized here because first propagation create weights in the
						// 'unlinked' mode
						tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
						consoleTextArea.appendText("weigts randomized [ " + tester.getInitWeightRange(0) + ", "
								+ tester.getInitWeightRange(1) + "] !" + "\n");
						consoleTextArea.appendText("modèle chargé avec succès. "+ "\n");
						Platform.runLater(() -> {
					    	consoleTextArea.setScrollTop(Double.MAX_VALUE); // Défiler automatiquement vers le bas
						});

					}

				}
			}
		});

		final TextField learningRateField = new TextField(String.valueOf(trainer.getLearningRate()));
		learningRateField.setPrefColumnCount(4);
		final TextField momentumField = new TextField(String.valueOf(trainer.getAlphaDeltaWeight()));
		momentumField.setPrefColumnCount(4);
		final Button submit = new Button("Apply");

		showLogs.setSelected(true);

		hbox1.setSpacing(10);
		hbox1.getChildren().addAll(train, trainStop, run, runTest, rand, clear, print, save, new Label("Learning rate"),
				learningRateField, new Label("Momentum"), momentumField, submit, showLogs);

		hbox2.setSpacing(10);
		hbox2.getChildren().addAll(comboSamples, load, printDeserializedNet);

		hbox1.setPadding(new Insets(5, 20, 2, 20));
		hbox2.setPadding(new Insets(2, 20, 5, 20));

		animation.setOnFinished(new EventHandler<ActionEvent>() {

			@Override
			public void handle(ActionEvent arg0) {
				Platform.runLater(() -> {
					train.setDisable(false);
				});
			}

		});
		
		train.setOnAction(new EventHandler<ActionEvent>() {

			@Override
			public void handle(ActionEvent e) {

				try {
					
					tabPane.getSelectionModel().select(tabTraining);
					
					trainer.getErrorLevelLines().clear();

					animation.stop(); // Arrêtez l'animation avant de commencer l'entraînement
					Platform.runLater(() -> {
						train.setDisable(true);
					});
		            // S'il n'y a pas encore de données, initialisez l'axe des X en fonction de la dernière valeur de cycle d'entraînement
		            if (lineChart.getData() == null || lineChart.getData().isEmpty()) {
		                xAxis.setAutoRanging(false); // Désactivez l'auto-range pour définir manuellement les limites
		                xAxis.setLowerBound(origTrainingCycle); // Démarrez à partir de la dernière valeur du cycle d'entraînement
		            }
	                xAxis.setUpperBound(lastTrainingCycle + nbCycles); // Exemple: +500 pour définir une nouvelle plage

		            if (series == null) {
		                series = new LineChart.Series<Number, Number>();
		                lineChart.getData().add(series);
		            }

		            series.setName("Train " + (lineChart.getData().size() + 1));

					trainer.launchTrain(showLogs.isSelected(), consoleTextArea);

					timeline1.setCycleCount(trainer.getErrorLevelLines().size());
					animation.play();
					lastTrainingCycle = lastTrainingCycle + nbCycles;

				} catch (Exception e1) {
					e1.printStackTrace();
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
								tabPane.getSelectionModel().select(tabConsole);
								consoleTextArea.setText("Concurrent modif KeyFrame :  nextIndex :" + nextIndex);
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
					train.setText("Train");
					factor = 0;
					Thread.sleep(500);

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

		// Setting an action for the Submit button
		submit.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				if ((learningRateField.getText() != null && !learningRateField.getText().isEmpty())) {

					trainer.setLearningRate(Double.valueOf(learningRateField.getText()));
					trainer.setAlphaDeltaWeight(Double.valueOf(momentumField.getText()));
					tabPane.getSelectionModel().select(tabConsole);
					consoleTextArea.setText("learningRate set to " + learningRateField.getText());
					consoleTextArea.appendText("momentum set to " + momentumField.getText());
				} else {
					consoleTextArea.setText("nothing happened !");
				}
			}
		});

		save.setOnAction(new EventHandler<ActionEvent>() {
		    @Override
		    public void handle(ActionEvent e) {
		        // Format pour le timestamp
		        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
		        // Obtenir le timestamp actuel
		        String timestamp = LocalDateTime.now().format(formatter);

		        // Chemin du dossier models à la racine du projet
		        String cheminDossier = "models/";
		        // Vérifier l'existence du dossier et le créer si nécessaire
		        File dossier = new File(cheminDossier);
		        if (!dossier.exists()) {
		            dossier.mkdirs();
		        }

		        // Nom du fichier avec timestamp
		        String nomFichier = "deeper-net-model-" + timestamp + ".ser";
		        // Chemin complet du fichier de sauvegarde
		        String cheminFichier = cheminDossier + nomFichier;

		        // Obtenir le réseau de neurones à sauvegarder
		        Object reseauNeurone = tester.getNetwork();

		        // Appeler saveNetwork pour sauvegarder le réseau
		        saveNetwork(reseauNeurone, cheminFichier);
		        
				tabPane.getSelectionModel().select(tabConsole);
				consoleTextArea.setText("Réseau sauvegardé sous : " + cheminFichier);

		    }
		});
		
		
		load.setOnAction(new EventHandler<ActionEvent>() {
		    @Override
		    public void handle(ActionEvent event) {
		        FileChooser fileChooser = new FileChooser();
		        // Définir le filtre d'extension pour ne montrer que les fichiers .ser
		        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("SER files (*.ser)", "*.ser");
		        fileChooser.getExtensionFilters().add(extFilter);

		        // Ouvrir la boîte de dialogue de sélection de fichier
		        File file = fileChooser.showOpenDialog(null);

		        if (file != null) {
		            // Charger le réseau de neurones depuis le fichier sélectionné

			        try {
			            Network network = NetworkService.loadNetwork(file.getPath());

			            if (network != null) {
			            	
			            	lineChart.setTitle(network.getName());
			            	
			                if (tester != null) {
			                	
				                // Update your application's state with the loaded network if necessary
				                // For example, set the loaded network as the current network in your application context
					            Network.network = network;
					            
			                	initFireTimeT(network);
			                	
			                    tester.setNetwork(network); // Suppose que tester a une méthode setNetwork pour configurer le réseau

				                // Assuming you have the network object loaded successfully
				                // Enable the Run and Print Network buttons here
				                run.setDisable(false);
				                print.setDisable(false);
				                rand.setDisable(false);
				                printDeserializedNet.setDisable(false);
								consoleTextArea.setText("Modèle \"" + network.getName() + "\" chargé avec succès.");
			                } else {
			                    // Gérer le cas où tester n'est pas initialisé
			                    // Peut-être initialiser tester ici ou loguer une erreur
								tabPane.getSelectionModel().select(tabConsole);
								consoleTextArea.setText("Le Testeur n'est pas initialisé.");
			                }

			            }
			            
			        } catch (Exception e) {
			            e.printStackTrace();
			            // You might want to keep the buttons disabled if loading fails
			            run.setDisable(true);
			            print.setDisable(true);
			        }
		        
		        }
		        
		        
		    }

			private void initFireTimeT(Network network) {
					network.getContext().setClock(-1);
				    for (Link link : network.getAllLinks()) {
				    	if(link != null)
				    		link.initFireTimes();
				    	//				        double currentFireTimeT = link.getFireTimeT();
//				        double delta = link.getDelay(); // Ou une méthode statique/variable globale, selon votre implémentation
//				        link.setFireTimeT(currentFireTimeT - delta);
				    }
			}
		});
		




		rand.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				tester.initWeights(tester.getInitWeightRange(0), tester.getInitWeightRange(1));
				tabPane.getSelectionModel().select(tabConsole);
				consoleTextArea.setText("weigts randomized [ " + tester.getInitWeightRange(0) + ", "
						+ tester.getInitWeightRange(1) + "] !");
			}
		});

		clear.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				origTrainingCycle = lastTrainingCycle;
				if (!lineChart.getData().isEmpty())
			           lineChart.getData().clear(); // Utilisez clear() au lieu de setData(null)
		        series = null; // Réinitialisez la série pour être sûrs qu'une nouvelle série sera créée lors du prochain entraînement
			}
		});

		print.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				String trace = tester.getNetwork().getString();
				tabPane.getSelectionModel().select(tabConsole);
				consoleTextArea.setText(trace); // Mettre à jour l'area de texte de la console avec les informations

			}
		});
		
		printDeserializedNet.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent e) {
				Object object = tester.getNetwork();
	            Set<Integer> seenObjects = new HashSet<>();
	            try {
					tabPane.getSelectionModel().select(tabConsole);
					consoleTextArea.setText(NetworkService.printObjectDetails(object, 0, seenObjects));
				} catch (IllegalAccessException e1) {
					e1.printStackTrace();
				}

			}
		});

		return commandPane;

	}

	public Pane createNetPane() {

		return new HBox();
	}

	public Pane createScaling() {

		final INetwork network = tester.getNetwork();

		scalingPane.setStyle("-fx-background-color: #ffffff;-fx-border-color: #ff0000;-fx-border-width: 1px;");
		scalingPane.getChildren().clear();
		VBox vbox = new VBox();

		Slider scaleNode = null;
		Label label = null;
		
		if(network != null) {
			
			for (int idxLayerHidden = 1; idxLayerHidden <= network.getLayers().size() - 2; idxLayerHidden++) {
				int nodecount = network.getLayer(idxLayerHidden).getNodeCount();
				scaleNode = new Slider(0D, nodecount * 10D + 1D, nodecount);
				double majTick = 100D;
				if(nodecount <= 10) majTick = 10; else if(nodecount > 10 && nodecount < 100) majTick = 100; else if(nodecount > 100 && nodecount < 1000) majTick = 1000; else majTick = 10000;
				label = new Label("Hidden layer #" + idxLayerHidden + " nodes: " + nodecount);
	
				scaleNode.setMinorTickCount(10);
				scaleNode.setMajorTickUnit(majTick);
				scaleNode.setSnapToTicks(false);
				scaleNode.setShowTickLabels(true);
				scaleNode.setShowTickMarks(true);
				scaleNode.setPrefWidth(scalingPane.getWidth() - 50D);
	
				final int currentIdxLayer = idxLayerHidden;
				scaleNode.valueProperty().addListener(new ChangeListener<Number>() {
					public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
	
						if (old_val == null || new_val == null)
							return;
	
	
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
								((IArea) node.getArea()).removeNode(node);
							}
	
						}
	
					}
				});
				vbox.setPadding(new Insets(15,15,15,15));
				vbox.getChildren().addAll(label, scaleNode);
			}
		
		}

		scalingPane.getChildren().add(vbox);

		return scalingPane;
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
	
	public void saveNetwork(Object reseauNeurone, String cheminFichier) {
	    try {
			NetworkService.saveNetwork(reseauNeurone, cheminFichier);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void loadNetwork(String cheminFichier) {
	    try {
			NetworkService.loadNetwork(cheminFichier);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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

	public static void addSeriesToLineChart() {
	}

	public static int getThreadPoolDelay() {
		return threadPoolDelay;
	}

	public static void setThreadPoolDelay(int threadPoolDelay) {
		ViewerFX.threadPoolDelay = threadPoolDelay;
	}

}