package RN.algotrainings;

import java.util.List;

import RN.ENetworkImplementation;
import RN.ILayer;
import RN.algoactivations.EActivation;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;

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
				
				if (layer.isLastLayer()) {
					

					
					// Pour la dernière couche avec cross-entropy + softmax
	                // Mettre à jour le taux d'erreur total pour l'évaluation de la performance du modèle
					 derivatedError = node.getError();
					 errorRate += -Math.log(Math.max(node.getComputedOutput(), 1e-15)) * node.getIdealOutput();

				} else {

					// somme pondérés du produit des poids des noeuds reliés sur
					// la couche précédente et de l'erreur du noeud reliés sur
					// la couche precedente
					if(getNetwork().getImpl() == ENetworkImplementation.LINKED){
						
						for (Link link : node.getOutputs()) {
							if(link != null && (link.getType() == ELinkType.REGULAR || link.getType() == ELinkType.SHARED)){
								derivatedError += link.getWeight() * getDerivatedErrorFromTargetNode(link);
							}
						}
						
					}else{
						
						// TODO Optimiser la methode
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
	
	


	private double getDerivatedErrorFromTargetNode(Link link) throws Exception {
		
		if(link.getTargetNode().getArea().getActivation() == EActivation.SOFTMAX) {
			return EActivation.getAreaPerformer(EActivation.SOFTMAX, link).performDerivative();
		} else {
			return link.getTargetNode().getDerivatedError();
		}
	}


	
}
