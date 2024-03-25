package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public class HessianCourbureLinkage extends FilterLinkage {

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
	


	public HessianCourbureLinkage() {
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
//		double k = params[0];
		
//		ox = (N-1) * deltaX / k;
//		ox = 0.5D * k;
//		oy = ox;
//		alpha = 1D;
//		mu = 0D;
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;

		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		FilterIndex index1 = new FilterIndex(((IPixelNode) thisNode).getAreaSquare().getIdentification(), ID_FILTER_Gxx);
		FilterIndex index2 = new FilterIndex(((IPixelNode) thisNode).getAreaSquare().getIdentification(), ID_FILTER_Gyy);
		FilterIndex index3 = new FilterIndex(((IPixelNode) thisNode).getAreaSquare().getIdentification(), ID_FILTER_Gxy);

		double divisor = 1D;
		double gxx = 0D;
		double gyy = 0D;
		double gxy = 0D;

		for (INode sourceNode : subArea.getNodes()) {

			gxx += getFilterValue(index1, EFilterPosition.CENTER, (IPixelNode) thisNode, (IPixelNode) sourceNode) * sourceNode.getComputedOutput();
			gyy += getFilterValue(index2, EFilterPosition.CENTER, (IPixelNode) thisNode, (IPixelNode) sourceNode) * sourceNode.getComputedOutput();
			gxy += getFilterValue(index3, EFilterPosition.CENTER, (IPixelNode) thisNode, (IPixelNode) sourceNode) * sourceNode.getComputedOutput();

		}

		sigmaWI = (gxx * gyy - Math.pow(gxy, 2D)) / Math.pow(gxx + gyy, 2D);

		// double r = 10D;
		// double R = r / Math.pow(r + 1, 2D);
		// System.out.println("criterion = " + (criterion *
		// interestPointNode.getComputedOutput()) + " >= " + R);
		//
		// sigmaWI -= thisNode.getBiasWeight();

		return sigmaWI;
	}
	

	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
//		double weight = 0D;
//		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
//		
//		thisNode.getArea().initFilter(this, ID_FILTER_GAUSSIAN, ESamples.GAUSSIAN_DE_MARR, subArea);
//		
//		for (INode sublayerNode : subArea.getNodes()) {
//			if (sublayerNode.getNodeType() == ENodeType.PIXEL) {
//				
//				// on connecte les neurones suivant la dérivée seconde de la gaussienne
//				// réalisant ainsi le filtre de Marr ou Laplacien de Gaussienne ou chapeau mexicain
//				weight = thisNode.getArea().getFilterValue(thisNode, ID_FILTER_GAUSSIAN, sublayerNode);
//				if(Math.abs(weight) > 0.001D){
//					sublayerNode.link(thisNode, ELinkType.REGULAR, isWeightModifiable(), weight);
//				}
//				
//			}
//		}
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

	public Double getOx() {
		return ox;
	}

	public void setOx(Double ox) {
		this.ox = ox;
	}

	public Double getOy() {
		return oy;
	}

	public void setOy(Double oy) {
		this.oy = oy;
	}

}
