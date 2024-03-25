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
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public class DifferenceOfGaussianLinkage extends FilterLinkage {
	
	// GAUSSIENNE ELLIPTIQUE
	private Double mu = null;

	// centre
	private Double x1 = null;
	private Double y1 = null;

	// amplitude, hauteur de la forme pixelisée
	private Double alpha1 = null;
	private Double alpha2 = null;

	// ecartement selon x et y
	private Double ox1 = null;
	private Double oy1 = null;
	
	private Double ox2 = null;
	private Double oy2 = null;
	
	// k un paramètre liant les variances des deux fonctions gaussiennes.
	private Double k0 = null;

	// Interface graphique pour ImageNode
	private  Slider kSlider = null;
	


	public DifferenceOfGaussianLinkage() {
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
		
		k0 = params[0];
		mu = 0D;
		
		initParametersGaussian1(k0);
		initParametersGaussian2();
		
	}

	private void initParametersGaussian1(Double k1) {
		
		ox1 = 0.5D;
		oy1 = ox1;
		alpha1 = 1D; //2.3D;
		
	}
	
	private void initParametersGaussian2() {
		
		ox2 = 0.5D;
		oy2 = ox2;
		alpha2 = 1D; //2.3D;
		
	}
	
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
				
		
		initFilter(this, ID_FILTER_DOG_0, ESamples.GAUSSIAN_DE_MARR, (IPixelNode)thisNode, subArea, getMu(), getAlpha1(), getOx1(), getOy1(), getK0());
		initFilter(this, ID_FILTER_DOG_1, ESamples.GAUSSIAN_DE_MARR, (IPixelNode)thisNode, subArea, getMu(), getAlpha2(), getOx2(), getOy2());
		
//			thisNode.getArea().initCompositeFilter(this, ID_FILTER_DOG_0, ESamples.SUBSTRACT, ESamples.GAUSSIAN_DOG_DE_MARR, ESamples.GAUSSIAN_DE_MARR, subArea, getMu(), getAlpha1(), getOx1(), getOy1(), getK0(), getMu(), getAlpha2(), getOx2(), getOy2(), getK1());

		subArea.applyConvolutionCompositeFilter(this, ID_FILTER_DOG_0, ID_FILTER_DOG_1, ESamples.SUBSTRACT, (IPixelNode) thisNode, sigmaWI);
			
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		initFilter(this, ID_FILTER_DOG_0, ESamples.GAUSSIAN_DE_MARR, (IPixelNode) thisNode, subArea, getMu(), getAlpha1(), getOx1(), getOy1(), getK0());
		initFilter(this, ID_FILTER_DOG_1, ESamples.GAUSSIAN_DE_MARR, (IPixelNode) thisNode, subArea, getMu(), getAlpha2(), getOx2(), getOy2());
		
		//thisNode.getArea().initCompositeFilter(this, ID_FILTER_DOG_0, ESamples.SUBSTRACT, ESamples.GAUSSIAN_DOG_DE_MARR, ESamples.GAUSSIAN_DE_MARR, subArea, getMu(), getAlpha1(), getOx1(), getOy1(), getK0(), getMu(), getAlpha2(), getOx2(), getOy2(), getK1());
		
		
		subArea.applyConvolutionCompositeFilter(this, ID_FILTER_DOG_0, ID_FILTER_DOG_1, ESamples.SUBSTRACT, (IPixelNode) thisNode, 0.001f);
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre gaussien
		return params[0] + params[1] * InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				(double) subArea.getHeightPx(), 
				(double) sublayerNode.getX(),
				(double) sublayerNode.getY(), 
				(double) subArea.getNodeCenterXY().getX(), 
				(double) subArea.getNodeCenterXY().getY(), 
				params[2], 
				params[3],
				params.length == 5 ? params[4] : null);
	}
	
	

	
	public void addGraphicInterface(Pane pane) {
		
		kSlider = new Slider(-2D, 2D, getK0());
		TextField label0 = new TextField("k=" + getK0());
		//TextField label1 = new TextField("k1=" + getK1());
		TextField sigma1 = new TextField("sigma1=" + getOx1());
		TextField sigma2 = new TextField("sigma2=" + getOx2());
		Button showFilterBtn = new Button("show filter");
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(kSlider);
		
		HBox hbox1 = new HBox(label0);
		HBox hbox2 = new HBox(sigma1, sigma2);
		HBox hbox3 = new HBox(showFilterBtn);
		
			
			
			showFilterBtn.setTooltip(new Tooltip("show filter"));
			
			
			showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

				@Override
				public void handle(ActionEvent event) {
					try {
						//FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_CONVOLUTION);
						((IAreaSquare)thisArea).getFilter(FilterLinkage.ID_FILTER_DOG_0).filterToImage(8);
						((IAreaSquare)thisArea).getFilter(FilterLinkage.ID_FILTER_DOG_1).filterToImage(8);
						//thisArea.showImageArea();
					} catch (Exception e) {
						e.printStackTrace();
					}
					
				}
				
			});
			

		pane.getChildren().addAll(hbox, hbox1, hbox2, hbox3);

		kSlider.setBlockIncrement(0.1);
		kSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
//				if(new_val != old_val){
//					try {
//						setK0(new_val.doubleValue());
//						initParametersGaussian1(getK0());
//						sigma1.setText("sigma1=" + getOx1());
//						label0.setText(new_val.toString());
//						FilterLinkage.removeFilter(new FilterIndex(ID_FILTER_DOG_0));
//						getArea().setFilter(ID_FILTER_DOG_1, null);
//						for (INode node : getArea().getNodes()) {
//							node.computeOutput(false);
//						}
//						getArea().showImageArea();
//						System.out.println("k: " +getK0());
//					} catch (Exception e) {
//						e.printStackTrace();
//					}
//				}
			}
		});
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

	public Double getK0() {
		return k0;
	}

	public void setK0(Double k) {
		this.k0 = k;
	}
	

	public Double getMu() {
		return mu;
	}

	public void setMu(Double mu) {
		this.mu = mu;
	}

	public Double getAlpha1() {
		return alpha1;
	}

	public void setAlpha1(Double alpha1) {
		this.alpha1 = alpha1;
	}

	public Double getAlpha2() {
		return alpha2;
	}

	public void setAlpha2(Double alpha2) {
		this.alpha2 = alpha2;
	}

	public Double getOx1() {
		return ox1;
	}

	public void setOx1(Double ox1) {
		this.ox1 = ox1;
	}

	public Double getOy1() {
		return oy1;
	}

	public void setOy1(Double oy1) {
		this.oy1 = oy1;
	}

	public Double getOx2() {
		return ox2;
	}

	public void setOx2(Double ox2) {
		this.ox2 = ox2;
	}

	public Double getOy2() {
		return oy2;
	}

	public void setOy2(Double oy2) {
		this.oy2 = oy2;
	}

	public Slider getkSlider() {
		return kSlider;
	}

	public void setkSlider(Slider kSlider) {
		this.kSlider = kSlider;
	}


}
