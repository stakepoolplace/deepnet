package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public class FirstDerivatedGaussianLinkage extends FilterLinkage {
	
	// GAUSSIENNE ELLIPTIQUE
	private Double mu = null;

	// centre
	private Double x1 = null;
	private Double y1 = null;

	// amplitude, hauteur de la forme pixelisée
	private Double alpha = null;

	// ecartement selon x et y
	private Double ox = null;
	private Double oy = null;

	

	public FirstDerivatedGaussianLinkage() {
	}
	
	public void initParameters() {
		
//		Le traitement repose sur cinq paramètres :
//		
//			N représente la taille du masque (matrice carrée) implantant le filtre LOG. N est impair.
//			σ permet d'ajuster la taille du chapeau mexicain.
//			∆x et ∆y sont les pas d'échantillonnage utilisés pour discrétiser h''(x,y). Généralement ∆x = ∆ y
//			S est le seuil qui permet de sélectionner les contours les plus marqués.
//		
//			Il est à noter que le choix des paramètres N, σ et ∆x ne doit pas se faire de façon indépendante. 
//  		En effet, le masque, même de taille réduite, doit ressembler à un chapeau mexicain. Le problème ici est le même que celui que l'on rencontre lors de l'échantillonnage d'une fonction gaussienne. 
//          Le nombre de points N à considérer doit être tel que l'étendue occupe l'intervalle [-3σ , 3σ].
//			En fonction du pas d'échantillonnage, l'étendue spatiale vaut : (N-1) ∆x  .
//			Cette étendue peut aussi s'écrire en fonction de σ : (N-1) ∆x = kσ  avec k entier.
//			En prenant par exemple  ∆x = 1 , il s'agit de choisir N et σ de sorte que l'étendue du chapeau mexicain soit pertinente. 
//  		Pour le chapeau mexicain, la valeur de k doit être au moins de 4.
		
		double N = 3;
		double deltaX = 1;
		double k = 4D;
		
		ox = (N-1) * deltaX / k;
		oy = ox;
		alpha = 1D;
		mu = 0D;
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
				
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
	
		initFilter(this, ID_FILTER_FIRST_DERIVATED_GAUSSIAN, ESamples.G_Dxy_DE_MARR, (IPixelNode) thisNode, subArea);
	
		subArea.applyConvolutionFilter(this, ID_FILTER_FIRST_DERIVATED_GAUSSIAN, (IPixelNode) thisNode, sigmaWI);

		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer previousLayer) {
		
			
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_FIRST_DERIVATED_GAUSSIAN, ESamples.G_Dxy_DE_MARR, (IPixelNode)thisNode, subArea);
		
		subArea.applyConvolutionFilter(this, ID_FILTER_FIRST_DERIVATED_GAUSSIAN, (IPixelNode)thisNode, 0.0001f);

	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre gaussien
		return mu + alpha * InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				(double) subArea.getHeightPx(), 
				(double) sublayerNode.getX(),
				(double) sublayerNode.getY(), 
				(double) subArea.getNodeCenterXY().getX(), 
				(double) subArea.getNodeCenterXY().getY(), 
				ox, 
				oy);
	}
	
	public void addGraphicInterface(Pane pane) {
		
		Button showFilterBtn = new Button("show filter");
		
		showFilterBtn.setTooltip(new Tooltip("show filter"));
		
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(showFilterBtn);
		
		showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

			@Override
			public void handle(ActionEvent event) {
				try {
					//FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_CONVOLUTION);
					((IAreaSquare)thisArea).getFilter(ID_FILTER_FIRST_DERIVATED_GAUSSIAN).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		
		pane.getChildren().addAll(hbox);
		
		
		
	}
	
	
	

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}

}
