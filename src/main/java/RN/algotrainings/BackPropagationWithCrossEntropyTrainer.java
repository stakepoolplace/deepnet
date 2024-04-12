package RN.algotrainings;

import java.util.List;

import RN.DataSeries;
import RN.ENetworkImplementation;
import RN.ILayer;
import RN.algoactivations.EActivation;
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
 * Backpropagation with Cross-entropy function error
 *
 */
public class BackPropagationWithCrossEntropyTrainer extends BackPropagationTrainer implements ITrainer {



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
				
				if (layer.isLastLayer() && node.getArea().getActivation() == EActivation.SOFTMAX) {
					
					// Pour la dernière couche avec cross-entropy + softmax
	                // Mettre à jour le taux d'erreur total pour l'évaluation de la performance du modèle
					derivatedError = node.getError();
					errorRate += -Math.log(Math.max(node.getComputedOutput(), 1e-15)) * node.getIdealOutput();

				} else if(layer.isLastLayer() && node.getArea().getActivation() != EActivation.SOFTMAX) {
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
				absoluteError = absoluteError / samplesNb;
				getNetwork().setAbsoluteError(absoluteError);

				// Sampling 1/10 de l'affichage de l'erreur
				if(maxTrainingCycles >= 100 && trainCycle % (maxTrainingCycles / 100) == 0)
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
	
	




	
}
