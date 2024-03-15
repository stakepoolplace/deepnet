package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

public class GaussianLinkage extends FilterLinkage {
	
	private  Slider sigmaSlider = null;

	// GAUSSIENNE ELLIPTIQUE
	private Double mu = null;

	// centre
	private Double x1 = null;
	private Double y1 = null;

	// amplitude, hauteur de la forme pixelisée
	private Double alpha = null;

	// ecartement selon x et y
	private Double sigmaX = null;
	private Double sigmaY = null;
	
	private Double k = null;
	


	public GaussianLinkage() {
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
		
//		double N = 9D;
//		double deltaX = 1D;
		k = params[0];
		
//		ox = (N-1) * deltaX / k;
		sigmaX = 0.5D;
		sigmaY = sigmaX;
		alpha = 1D;
		mu = 0D;
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_GAUSSIAN, ESamples.GAUSSIAN_DE_MARR, (IPixelNode) thisNode, subArea, getMu(), getAlpha(), getSigmaX(), getSigmaY(), getK());
		
		subArea.applyConvolutionFilter(this, ID_FILTER_GAUSSIAN, (IPixelNode) thisNode, sigmaWI);
			
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
			IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			
			initFilter(this, ID_FILTER_GAUSSIAN, ESamples.GAUSSIAN_DE_MARR, (IPixelNode) thisNode, subArea, getMu(), getAlpha(), getSigmaX(), getSigmaY(), getK());
			
			subArea.applyConvolutionFilter(this, ID_FILTER_GAUSSIAN, (IPixelNode)thisNode, 0.001f);
		
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
				sigmaX, 
				sigmaY,
				k);
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
	
	public void addGraphicInterface(Pane pane) {
		
		sigmaSlider = new Slider(0D, 5D, 0.5D);
		sigmaSlider.setBlockIncrement(0.1D);
		sigmaSlider.setTooltip(new Tooltip("Sigma"));
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(sigmaSlider);
		
		
		sigmaSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setSigmaX(new_val.doubleValue());
				System.out.println(String.format("Sigma : %.2f", new_val));
				try {
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GAUSSIAN);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		Button showFilterBtn = new Button("show filter");
		
		showFilterBtn.setTooltip(new Tooltip("show filter"));
		
		
		HBox hbox2 = new HBox();
		hbox2.getChildren().addAll(showFilterBtn);
		
		showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

			@Override
			public void handle(ActionEvent event) {
				try {
					//FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_CONVOLUTION);
					((IAreaSquare)thisArea).getFilter(ID_FILTER_GAUSSIAN).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		

		pane.getChildren().addAll(hbox, hbox2);
		
	}

	public Double getMu() {
		return mu;
	}

	public void setMu(Double mu) {
		this.mu = mu;
	}

	public Double getX1() {
		return x1;
	}

	public void setX1(Double x1) {
		this.x1 = x1;
	}

	public Double getY1() {
		return y1;
	}

	public void setY1(Double y1) {
		this.y1 = y1;
	}

	public Double getAlpha() {
		return alpha;
	}

	public void setAlpha(Double alpha) {
		this.alpha = alpha;
	}

	public Double getSigmaX() {
		return sigmaX;
	}

	public Double getK() {
		return k;
	}

	public void setK(Double k) {
		this.k = k;
	}

	public Slider getSigmaSlider() {
		return sigmaSlider;
	}

	public void setSigmaSlider(Slider sigmaSlider) {
		this.sigmaSlider = sigmaSlider;
	}

	public Double getSigmaY() {
		return sigmaY;
	}

	public void setSigmaY(Double sigmaY) {
		this.sigmaY = sigmaY;
	}

	public void setSigmaX(Double sigmaX) {
		this.sigmaX = sigmaX;
	}

}
