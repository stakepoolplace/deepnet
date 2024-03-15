package RN.linkage;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Optional;

import RN.AreaSquare;
import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.linkage.vision.EKeyPointType;
import RN.linkage.vision.Gradient;
import RN.linkage.vision.Histogram;
import RN.linkage.vision.KeyPoint;
import RN.linkage.vision.KeyPointDescriptor;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;
import javafx.scene.layout.Pane;

public class OneToOneOctaveAreaLinkage extends FilterLinkage {

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
	
	// seuil de suppression des pixels de bas contraste
	private double C = 0D;
	
	private static int maxSuppressed = 0;
	private static int minSuppressed = 0;
	
	private List<KeyPoint> keyPoints = new ArrayList<KeyPoint>();
	
	public OneToOneOctaveAreaLinkage() {
	}
	
	public void prePropagation(){
		keyPoints.clear();
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
		
//		ox = (N-1) * deltaX / k;
		ox = 0.5D;
		oy = ox;
		alpha = 1D;
		mu = 0D;
		
	}
	
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		List<KeyPoint> maxPointInterest = null;
		List<KeyPoint> minPointInterest = null;
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		INode sourceNode = null;
		Double maxValue = 0D;
					
		maxPointInterest = maxLocal(subArea, (IPixelNode) thisNode);
		if(maxPointInterest != null){
			
			for(KeyPoint maxKP : maxPointInterest){
				
				maxValue = Math.max(maxValue, maxKP.getValue());
				System.out.println(maxKP);
				keyPoints.add(maxKP);
			}
			
			sourceNode = (INode) ((AreaSquare) subArea).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling);
			sigmaWI += maxValue * Linkage.getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
			
		}
		
		Double minValue = 0D;
		minPointInterest = minLocal(subArea, (IPixelNode) thisNode);
		if(minPointInterest != null){

			for(KeyPoint minKP : minPointInterest){
				
				minValue = Math.min(minValue, minKP.getValue());
				System.out.println(minKP);
				keyPoints.add(minKP);
			}
			
			sourceNode = (INode) ((AreaSquare) subArea).getNodeXY(((IPixelNode) thisNode).getX(), ((IPixelNode) thisNode).getY(), sampling);
			sigmaWI += minValue * Linkage.getLinkAndPutIfAbsent(thisNode, sourceNode, isWeightModifiable()).getWeight();
			
		}
							
				
			
		if(thisNode.getBiasWeightValue() != null)
			sigmaWI -= thisNode.getBiasWeightValue();
		
		return sigmaWI;
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer previousLayer) {
		
		List<KeyPoint> maxPointInterest = null;
		List<KeyPoint> minPointInterest = null;
			
//			for(IArea area : getLinkedAreas()){
//				
//					if(area.getAreaId() > 0 && area.getAreaId() < sublayer.getAreas().size() - 1){
//							
//						maxPointInterest = maxLocal(area, (IPixelNode) thisNode);
//						if(maxPointInterest != null){
//							for(KeyPoint maxKP : maxPointInterest){
//								maxKP.getKeyPointNode().link(thisNode, ELinkType.REGULAR, false, maxKP.getValue());
//							}
//						}
//						
//						minPointInterest = minLocal(area, (IPixelNode) thisNode);
//						if(minPointInterest != null){
//							for(KeyPoint minKP : minPointInterest){
//								minKP.getKeyPointNode().link(thisNode, ELinkType.REGULAR, false, minKP.getValue());
////								sigmaWI += Math.abs(minKP.getValue());
//							}
//						}
//							
//					}
//				
//			}
			
		
	}
	
	
	
	public List<KeyPoint> maxLocal(IAreaSquare area, IPixelNode thisNode){
		
		IPixelNode interestPointNode =  area.getNodeXY(thisNode.getX(), thisNode.getY());
		IAreaSquare gaussianArea = (IAreaSquare) interestPointNode.getAreaSquare().getLeftSibilingArea();
		
		boolean isMax = true;
		
//		if(interestPointNode.getComputedOutput() < C)
//			return null;
		
		if(interestPointNode.getPreviousAreaSquare() != null){
			isMax &= maxLocalOnArea(interestPointNode, interestPointNode.getPreviousAreaSquare().getNode(interestPointNode.getNodeId()));
			if(!isMax)
				return null;
		}
		
		isMax &= maxLocalOnArea(interestPointNode, interestPointNode);
		if(!isMax)
			return null;
		
		if(interestPointNode.getNextAreaSquare() != null){
			isMax &= maxLocalOnArea(interestPointNode, interestPointNode.getNextAreaSquare().getNode(interestPointNode.getNodeId()));
			if(!isMax)
				return null;
		}
		
		if(!hessianCriterion(thisNode, interestPointNode)){
			maxSuppressed++;
			return null;
		}
		
		List<KeyPoint> maxKeyPoints = computeSegmentedOrientationsAtKeyPoint(EKeyPointType.MAX, interestPointNode, gaussianArea);
		
		computeDescriptors(maxKeyPoints);
		
		return maxKeyPoints;
		
	}
	
	public List<KeyPoint> minLocal(IAreaSquare area, IPixelNode thisNode){
		
		boolean isMin = true;
		
		IPixelNode interestPointNode =  area.getNodeXY(thisNode.getX(), thisNode.getY());
		IAreaSquare gaussianArea = (IAreaSquare) interestPointNode.getAreaSquare().getLeftSibilingArea();
		
//		if(interestPointNode.getComputedOutput() < C)
//			return null;
		
		if(interestPointNode.getPreviousAreaSquare() != null){
			isMin &= minLocalOnSameArea(interestPointNode, interestPointNode.getPreviousAreaSquare().getNode(interestPointNode.getNodeId()));
			if(!isMin)
				return null;
		}
		
		
		isMin &= minLocalOnSameArea(interestPointNode, interestPointNode);
		if(!isMin)
			return null;
		
		
		if(interestPointNode.getNextAreaSquare() != null){
			isMin &= minLocalOnSameArea(interestPointNode, interestPointNode.getNextAreaSquare().getNode(interestPointNode.getNodeId()));
			if(!isMin)
				return null;
		}
		
		if(!hessianCriterion(thisNode, interestPointNode)){
			minSuppressed++;
			return null;
		}
		
		List<KeyPoint> minKeyPoints = computeSegmentedOrientationsAtKeyPoint(EKeyPointType.MIN, interestPointNode, gaussianArea);
		
		computeDescriptors(minKeyPoints);
		
		return minKeyPoints;
		
		
	}
	

	private List<KeyPointDescriptor> computeDescriptors(List<KeyPoint> keyPoints) {
		
		List<KeyPointDescriptor> descriptors = new ArrayList<KeyPointDescriptor>();
		
		KeyPointDescriptor descriptor = null;
		for(KeyPoint kp : keyPoints){
			
			// Rotation de l'axe des absisses sur le gradient du point d'interet
			// Calcul des gradients dans chacune des cellules de la zone des 16x16 pixels (+1 colonne centrale).
			Gradient[][] gradients = computeRotatedGridGradients(kp, 17);
			
			// Magnitude Threshold at 0.1 times the max for illumination stability
			gradientThreshold(gradients);
	
			// Création d'une grille 4x4 autour du point d'interet.
			// Calcul d'histogrammes de 9 orientations dans chacune des cellules.
			// Normalisation des valeurs des gradients.
			// Création des vecteurs descripteurs de 4x4x8 dimensions par point d'interet.
			descriptor = computeGridXBinsHistograms(kp, gradients, 9, 17, 4);
			kp.setDescriptor(descriptor);
			descriptors.add(descriptor);
			
			// Normalisation des descripteurs.
			normalizeDescriptor(kp);
			
		}
		
		return descriptors;
		
	}


	private void normalizeDescriptor(KeyPoint kp) {

		KeyPointDescriptor descriptor = kp.getDescriptor();

		for (Histogram histogram : descriptor.getHistograms()) {

			if (histogram.isEmpty()) {
				//histogram.put(0D, 0D);
			} else {
				for (Entry<Double, Double> entry : histogram.entrySet()) {
					entry.setValue(kp.normalizeValue(entry.getValue()));
				}
			}
		}

	}


	private KeyPointDescriptor computeGridXBinsHistograms(KeyPoint kp, Gradient[][] gradients, int binCount, int matrixOrder, int cellSizeOrder) {
		
		KeyPointDescriptor descriptor = new KeyPointDescriptor();
		
		// TODO Change it to 28 foveal gaussian overlapping zones
		// Blocks' iteration.
		// Each block point's starts at top left.
		for(int cellY = 0; cellY < matrixOrder - 1; cellY += cellSizeOrder){
			for(int cellX = 0; cellX < matrixOrder - 1; cellX += cellSizeOrder){
				
				try {
					descriptor.concatenateHistogram(computeHistogram(kp, binCount, getGradientsInSquarredOverlappingZone(cellX, cellY, cellSizeOrder, gradients)));
				} catch (Exception e) {
					e.printStackTrace();
					System.err.println("Calcul de l'histogramme 8 bins impossible pour : " + kp + " cellX : " + cellX + "  cellY : " + cellY);
				}
				
			}
		
		}
		
		return descriptor;
		
	}
	
	
	private Histogram computeHistogram(KeyPoint kp, int binsCount, List<Gradient> gradientsInOverlappingBlock) {
		
		Histogram histogram = new Histogram();
		Double value = null;
		
		for (Gradient gradient : gradientsInOverlappingBlock) {
				
				Double segmentedIdx = getSegmentedAngle(gradient.getTheta(), (2D * Math.PI) / binsCount);
				
				value = gradient.getMagnitude(); // * gradient.getDistanceToKeyPoint();// * kp.getKeyPointNode().getComputedOutput();
				
				// Normalize vote with small constant
				value = kp.normalizeValue(value);
				
				histogram.putIfAbsent(segmentedIdx, 0D);
				histogram.put(segmentedIdx, histogram.get(segmentedIdx) + value);
		}
		
		return histogram;
	}
	
	
	
	/**
	 * Return nodes in a circlar zone
	 * @param x0 x center point
	 * @param y0 y center point
	 * @param radius
	 * @return List<Gradient>
	 * @throws Exception
	 */
	public List<Gradient> getGradientsInCircularOverlappingZone(int x0, int y0, int radius, Gradient[][] gradients) {
		List<Gradient> gradientList = new ArrayList<Gradient>();
		for(int y = y0 - radius; y < y0 + radius; y++){
			for(int x = x0 - radius; x < x0 + radius; x++){
				double distance = Math.sqrt(Math.pow(x0 - x, 2D) + Math.pow(y0 - y, 2D));
				if(distance <= radius){
					try {
						if(gradients[x][y] != null)
							gradientList.add(gradients[x][y]);
					} catch (Exception ignore) {
						//e.printStackTrace();
						//System.err.println("getGradientsInCircularOverlappingZone() : Impossible de recuperer le gradient pour (x,y) = " + x + "," + y + ")" );
					}
				}
			}
		}
		return gradientList;
	}
	
	
	/**
	 * Return nodes in a squarred zone
	 * @param x0 x center point
	 * @param y0 y center point
	 * @param radius
	 * @return List<Gradient>
	 * @throws Exception
	 */
	public List<Gradient> getGradientsInSquarredOverlappingZone(int x0, int y0, int width, Gradient[][] gradients) {
		
		List<Gradient> gradientList = new ArrayList<Gradient>();
		
		// width + overlapping
		for(int y = -1; y < width + 1; y++){
			for(int x = -1; x < width + 1; x++){
					try {
						if(gradients[x0 + x][y0 + y] != null)
							gradientList.add(gradients[x0 + x][y0 + y]);
					} catch (Exception ignore) {
						//e.printStackTrace();
						//System.err.println("getGradientsInCircularOverlappingZone() : Impossible de recuperer le gradient pour (x,y) = " + x + "," + y + ")" );
					}
			}
		}
		return gradientList;
	}


	private Gradient[][] computeRotatedGridGradients(KeyPoint kp, int matrixOrder) {
		
			PixelNode neighbor = null;
			Gradient[][] gradients = new Gradient[matrixOrder][matrixOrder];
			int halfWidth = matrixOrder / 2;
			Gradient gradient = null;
			for(int y = -halfWidth; y <= halfWidth; y++){
				for(int x = -halfWidth; x <= halfWidth; x++){
					
					neighbor = (PixelNode) kp.getKeyPointNode().getAreaSquare().getNodeXY(x + kp.getX().intValue(), y + kp.getY().intValue(), kp.getX().intValue(), kp.getY().intValue(), kp.getTheta());
					
					if(neighbor == null)
						continue;
					
					gradient = kp.getGradient(neighbor.getX() - kp.getX().intValue() + halfWidth, neighbor.getY() - kp.getY().intValue() + halfWidth); 
					
					if(gradient == null)
						continue;
					
					gradient.setTheta(gradient.getTheta() - kp.getTheta());
					while(gradient.getTheta() < 0D){
						gradient.setTheta(gradient.getTheta() + (2D * Math.PI));
					}
					
					gradient.produceGradient(5D, null);
					gradients[x + halfWidth][y + halfWidth] = gradient;
					
				}
			}
			
			return gradients;
	}

	// TODO
	// 1.6D / Math.sqrt(D) with D is the descriptor dimensionality 
	private void gradientThreshold(Gradient[][] gradients, double threshold) {
		for(int idy = 0; idy < gradients.length; idy++){
			for(int idx = 0; idx < gradients[idy].length; idx++){
				gradients[idx][idy].thresholdAt(threshold);
			}
		}
	}
	
	private void gradientThreshold(Gradient[][] gradients) {
		for(int idy = 0; idy < gradients.length; idy++){
			for(int idx = 0; idx < gradients[idy].length; idx++){
				if(gradients[idx][idy] != null){
					gradients[idx][idy].thresholdAtMax(0.1D);
				}
			}
		}
	}
	
	
	
	
	private Gradient[][] computeGradients(IPixelNode interestPtNode, IAreaSquare gradientArea) {
		
		// Gradients on G * I
		Gradient gradient = null;
		GaussianLinkage dogLinkage = (GaussianLinkage) gradientArea.getLinkage();
		IPixelNode centerKpNode = gradientArea.getNodeXY(interestPtNode.getX(), interestPtNode.getY());
		
		// Sigma is 3 times that of the current smoothing scale.
		// More over, we use the calcul of the FWHM to evaluate the radius at 1/1000 from 0 on y.
		// y= σ * sqrt{2ln(1000)} + x_{center}
		// radius = 3 * σ * sqrt{2ln(1000)} + x_{center}
		Long radius = Math.round(dogLinkage.getK() * dogLinkage.getSigmaX() * 11.1501D);
		
		
		int gradientWidth = 2*radius.intValue();
		Gradient[][] gradients = new Gradient[gradientWidth][gradientWidth];

		List<IPixelNode> neighborhood = gradientArea.getNodesInCirclarZone(centerKpNode.getX(), centerKpNode.getY(), radius.intValue());
		//gradientArea.pixelsToString(neighborhood);
		
		IPixelNode pix = null;
		int idx;
		int idy;
		for (IPixelNode neighbor : neighborhood) {
				
				pix = (IPixelNode) neighbor;
				gradient = computeGradient(centerKpNode, neighbor);
				
				if(gradient != null){
					
					idx = pix.getX() - centerKpNode.getX() + radius.intValue();
					idy = pix.getY() - centerKpNode.getY() + radius.intValue();
					
					if(idx >= 0 && idx < gradientWidth && idy >= 0 && idy < gradientWidth)
						gradients[idx][idy] = gradient;
				}
				
		}
		
		// Magnitude Threshold at 0.1 times the max for illumination stability
		gradientThreshold(gradients);
		
		return gradients;
}


	private List<KeyPoint> computeSegmentedOrientationsAtKeyPoint(EKeyPointType kpType, IPixelNode interestPointNode, IAreaSquare gaussianArea) {
		
		List<KeyPoint> localKeyPoints = new ArrayList<KeyPoint>();
		Histogram histogram = new Histogram();
		double f = 1.5D;
		Double magnitude = null;
		Double segmentedIdx = null;
		
		// Calcul d'un histogramme sur une zone circulaire gaussienne
		Gradient[][] gradients = computeGradients(interestPointNode, gaussianArea);
		
		for(Gradient[] gradientRow : gradients){
			for(Gradient gradient : gradientRow){
				
				if(gradient == null)
					continue;
				
				magnitude = gradient.getMagnitude();// * gradient.getDistanceToKeyPoint() ; 
				
				segmentedIdx = getSegmentedAngle(gradient.getTheta(), (2D * Math.PI) / 36D);
				
				histogram.putIfAbsent(segmentedIdx, 0D);
				histogram.put(segmentedIdx, histogram.get(segmentedIdx) + magnitude); 
			}
		}
		
		Comparator<Double> comp = (p1, p2) -> Double.compare( p1.doubleValue(), p2.doubleValue());
		Optional<Double> magnitudeMax = histogram.values().stream().max(comp);
		
		if(magnitudeMax.isPresent()){
			
			// Suppression des orientations < 80% du max et création des points d'interets ainsi que les supplémentaires.
			GaussianLinkage dogLinkage = (GaussianLinkage) gaussianArea.getLinkage();
			
			Iterator<Entry<Double, Double>> it = histogram.entrySet().iterator();
			KeyPoint kp = null;
			Double normalizedMagnitude = null;
			while (it.hasNext())
			{
			   Entry<Double, Double> entry = it.next();
			   normalizedMagnitude = entry.getValue() / magnitudeMax.get();
				if(normalizedMagnitude < 0.8D){
					it.remove();
				}else{
					kp = new KeyPoint(kpType, interestPointNode, (double) interestPointNode.getX(), (double) interestPointNode.getY(), dogLinkage.getK() * dogLinkage.getSigmaX() * f, entry.getKey(), interestPointNode.getComputedOutput());
					kp.setGradients(gradients);
					localKeyPoints.add(kp);
				}
			}
		
		}
		
		return localKeyPoints;
		
	}
	
	private Double distance(IPixelNode n1, IPixelNode n2){
		
		return Math.sqrt(Math.pow(n1.getX() - n2.getX(), 2D) + Math.pow(n1.getY() - n2.getY(), 2D));
		
	}
	
	private Double getSegmentedAngle(double theta, double deltaTheta){
		
		return Math.floor(theta / deltaTheta) * deltaTheta;
	}
	

	private Gradient computeGradient(IPixelNode kp, IPixelNode neighboor) {
		
		Gradient gradient = null;
		
		try{
			double magnitude = Math.sqrt(
					Math.pow(neighboor.getLeft().getComputedOutput() - neighboor.getRight().getComputedOutput() , 2D) +
					Math.pow(neighboor.getDown().getComputedOutput() - neighboor.getUp().getComputedOutput() , 2D)
					);
			
			double theta = Math.atan2(
					(neighboor.getUp().getComputedOutput() - neighboor.getDown().getComputedOutput()),
					(neighboor.getLeft().getComputedOutput() - neighboor.getRight().getComputedOutput())
					);
			
			while(theta < 0D){
				theta += (2D * Math.PI);
			}
			
			double distanceToKeyPoint = distance(kp, neighboor);
			
			gradient = new Gradient(neighboor, theta, magnitude, distanceToKeyPoint);
			
			//gradient.produceGradient();
			
			return gradient;
			
		}catch(Throwable ignore){
			//ignore.printStackTrace();
		}
		
		return null;
		
	}


	private boolean maxLocalOnArea(IPixelNode refNode, IPixelNode localNode) {

		boolean isMax = true;

		if (localNode.getLeft() != null) {
			isMax &= refNode.compareOutputTo(localNode.getLeft()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getRight() != null) {
			isMax &= refNode.compareOutputTo(localNode.getRight()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getUp() != null) {
			isMax &= refNode.compareOutputTo(localNode.getUp()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getUpLeft() != null) {
			isMax &= refNode.compareOutputTo(localNode.getUpLeft()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getUpRight() != null) {
			isMax &= refNode.compareOutputTo(localNode.getUpRight()) > C;
			if (!isMax)
				return isMax;
		}
		if (localNode.getDown() != null) {
			isMax &= refNode.compareOutputTo(localNode.getDown()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getDownLeft() != null) {
			isMax &= refNode.compareOutputTo(localNode.getDownLeft()) > C;
			if (!isMax)
				return isMax;
		}

		if (localNode.getDownRight() != null) {
			isMax &= refNode.compareOutputTo(localNode.getDownRight()) > C;
			if (!isMax)
				return isMax;
		}

		if (refNode.getAreaSquare() != localNode.getAreaSquare()) {
			isMax &= refNode.compareOutputTo(localNode) > C;
			if (!isMax)
				return isMax;
		}

		return isMax;
	}
	
	private boolean minLocalOnSameArea(IPixelNode refNode, IPixelNode localNode) {

		boolean isMin = true;

		if (localNode.getLeft() != null) {
			isMin &= refNode.compareOutputTo(localNode.getLeft()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getRight() != null) {
			isMin &= refNode.compareOutputTo(localNode.getRight()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getUp() != null) {
			isMin &= refNode.compareOutputTo(localNode.getUp()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getUpLeft() != null) {
			isMin &= refNode.compareOutputTo(localNode.getUpLeft()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getUpRight() != null) {
			isMin &= refNode.compareOutputTo(localNode.getUpRight()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getDown() != null) {
			isMin &= refNode.compareOutputTo(localNode.getDown()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getDownLeft() != null) {
			isMin &= refNode.compareOutputTo(localNode.getDownLeft()) < C;
			if (!isMin)
				return isMin;
		}

		if (localNode.getDownRight() != null) {
			isMin &= refNode.compareOutputTo(localNode.getDownRight()) < C;
			if (!isMin)
				return isMin;
		}

		if (refNode.getAreaSquare() != localNode.getAreaSquare()) {
			isMin &= refNode.compareOutputTo(localNode) < C;
			if (!isMin)
				return isMin;
		}

		return isMin;
	}
	
	
	private boolean hessianCriterion(IPixelNode thisNode, IPixelNode interestPointNode) {
		
		double criterion = 0D;
		double divisor = 1D;
		double gxx = 0D;
		double gyy = 0D;
		double gxy = 0D;
		IAreaSquare area = interestPointNode.getAreaSquare();
		
		int x = ((PixelNode) interestPointNode).getX();
		int y = ((PixelNode) interestPointNode).getY();

		if(x > 0 && y > 0 && x < area.getWidthPx() - 1 && y < area.getHeightPx() - 1){
			
			gxx = area.getNodeXY(x + 1, y).getComputedOutput() - 2D * area.getNodeXY(x, y).getComputedOutput() + area.getNodeXY(x - 1, y).getComputedOutput();
			gyy = area.getNodeXY(x, y + 1).getComputedOutput() - 2D * area.getNodeXY(x, y).getComputedOutput() + area.getNodeXY(x, y - 1).getComputedOutput();
			gxy = (area.getNodeXY(x + 1, y + 1).getComputedOutput() 
					- area.getNodeXY(x + 1, y - 1).getComputedOutput() 
					- area.getNodeXY(x - 1, y + 1).getComputedOutput() 
					+ area.getNodeXY(x - 1, y - 1).getComputedOutput()) / 4D;
			
			
			divisor = Math.pow(gxx + gyy, 2D);
			
			if(divisor != 0D){
				criterion =  (gxx * gyy - Math.pow(gxy, 2D)) / divisor ;
			}
			
			double r = 10D;
			double R =  r / Math.pow(r + 1, 2D);
			//System.out.println("criterion = " + (criterion * interestPointNode.getComputedOutput()) + " >= " + R);
			return criterion  >= R;
		}
		
		return true;
	}
	

	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = sublayerNode.getAreaSquare();
		
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
	
	public void addGraphicInterface(Pane pane) {
		
		for(KeyPoint point : keyPoints){
			//System.out.println(point);
			point.produceCircle();
		}
		
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


	public List<KeyPoint> getKeyPoints() {
		return keyPoints;
	}


	public void setKeyPoints(List<KeyPoint> keyPoints) {
		this.keyPoints = keyPoints;
	}




}
