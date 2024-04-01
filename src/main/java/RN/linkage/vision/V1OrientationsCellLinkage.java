package RN.linkage.vision;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.linkage.FilterLinkage;
import RN.linkage.SigmaWi;
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

/**
 * @author Eric Marchand
 *
 */
public class V1OrientationsCellLinkage  extends FilterLinkage{

	private  Slider sigmaSlider = null;
	private  Slider thetaSlider = null;
	
	
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
	
	// Angle de détection des contours pour les nodes appartenant à cette zone. (cf. Vision aire V1)
	protected Double theta = null;
	
	protected Double k = null;
	
	

	public V1OrientationsCellLinkage() {
	}
	
	public void initParameters() {
		
//		Le traitement repose sur cinq paramètres :
//		
//			N représente la taille du masque (matrice carrée) implantant le filtre LOG. N est impair.
//			σ permet d'ajuster la taille du chapeau mexicain.
//			∆x et ∆y sont les pas d'échantillonnage utilisés pour discrétiser h''(x,y). Généralement ∆x = ∆y
//			S est le seuil qui permet de sélectionner les contours les plus marqués.
//		
//			Il est à noter que le choix des paramètres N, σ et ∆x ne doit pas se faire de façon indépendante. 
//  		En effet, le masque, même de taille réduite, doit ressembler à un chapeau mexicain. Le problème ici est le même que celui que l'on rencontre lors de l'échantillonnage d'une fonction gaussienne. 
//          Le nombre de points N à considérer doit être tel que l'étendue occupe l'intervalle [-3σ , 3σ].
//			En fonction du pas d'échantillonnage, l'étendue spatiale vaut : (N-1) ∆x  .
//			Cette étendue peut aussi s'écrire en fonction de σ : (N-1) ∆x = kσ  avec k entier.
//			En prenant par exemple  ∆x = 1 , il s'agit de choisir N et σ de sorte que l'étendue du chapeau mexicain soit pertinente. 
//  		Pour le chapeau mexicain, la valeur de k doit être au moins de 4.
		
		double N = 13;
		double deltaX = 1;
		double k = 14D;
		
		sigmaX = (N-1) * deltaX / k;
		
		setTheta(params[0]);
		
		if(params.length == 2){
			setSigmaX(params[1]);
		}
		
		if(params.length == 3){
			setK(params[2]);
		}
		
		sigmaY = sigmaX;
		alpha = 1D;
		mu = 0D;
		
		
		
	}

	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
			
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		
		if(getTheta() == null)
			throw new RuntimeException("Theta is not set on area");
		
		initFilter(this, ID_FILTER_V1Orientation, ESamples.G_D2xyTheta_DE_MARR, (IPixelNode) thisNode, subArea, getTheta(), getSigmaX());
		subArea.applyConvolutionFilter(this, ID_FILTER_V1Orientation, (IPixelNode) thisNode, sigmaWI);
			
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		if(getTheta() == null)
			throw new RuntimeException("Theta is not set on area");
		
		initFilter(this, ID_FILTER_V1Orientation, ESamples.G_D2xyTheta_DE_MARR, (IPixelNode) thisNode, subArea, getTheta(), getSigmaX());
		subArea.applyConvolutionFilter(this, ID_FILTER_V1Orientation, (IPixelNode) thisNode, 0.00000001f);		

		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		// Calcul du filtre gaussien
		// + Theta (angle du filtre en degré)
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				(double) subArea.getHeightPx(), 
				(double) sublayerNode.getX(),
				(double) sublayerNode.getY(), 
				(double) subArea.getNodeCenterX(), 
				(double) subArea.getNodeCenterY(),
				sigmaX,
				sigmaY,
				params[0]);
	}
	
	public void addGraphicInterface(Pane pane) {
		
		sigmaSlider = new Slider(0D, 5D, sigmaX);
		sigmaSlider.setBlockIncrement(0.1D);
		sigmaSlider.setTooltip(new Tooltip("Sigma"));
		
		thetaSlider = new Slider(0D, 360, theta);
		thetaSlider.setBlockIncrement(1D);
		thetaSlider.setTooltip(new Tooltip("Theta (degrees)"));
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(sigmaSlider, thetaSlider);
		
		
		sigmaSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setSigmaX(new_val.doubleValue());
				setSigmaY(new_val.doubleValue());
				System.out.println(String.format("Sigma : %.2f", new_val));
				try {
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_V1Orientation);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		thetaSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setTheta(new_val.doubleValue());
				System.out.println(String.format("Theta : %.2f", new_val));
				try {
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_V1Orientation);
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
					((IAreaSquare)thisArea).getFilter(ID_FILTER_V1Orientation).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		

		pane.getChildren().addAll(hbox, hbox2);
		
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

	public Double getTheta() {
		return theta;
	}

	public void setTheta(Double theta) {
		this.theta = theta;
	}

	public Double getK() {
		return k;
	}

	public void setK(Double k) {
		this.k = k;
	}

	public Double getSigmaX() {
		return sigmaX;
	}

	public void setSigmaX(Double sigmaX) {
		this.sigmaX = sigmaX;
	}

	public Double getSigmaY() {
		return sigmaY;
	}

	public void setSigmaY(Double sigmaY) {
		this.sigmaY = sigmaY;
	}

}
