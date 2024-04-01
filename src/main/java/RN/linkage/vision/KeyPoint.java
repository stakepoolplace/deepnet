package RN.linkage.vision;

import java.util.Map.Entry;

import RN.nodes.IPixelNode;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * @author Eric Marchand
 *
 */
public class KeyPoint {
	
	private IPixelNode keyPointNode = null;
	
	private EKeyPointType kpType = null;
	
	private Double x = null;
	
	private Double y = null;
	
	private Double sigma = null;
	
	private Double theta = null;
	
	private Double value = null;
	
	private Gradient[][] gradients = null;
	
	private KeyPointDescriptor descriptor = new KeyPointDescriptor();
	
	public KeyPoint(){
	}
	
	
	public KeyPoint(EKeyPointType kpType, IPixelNode keyPointNode, Double x, Double y, Double sigma, Double theta, Double value) {
		this.kpType = kpType;
		this.keyPointNode = keyPointNode;
		this.x = x;
		this.y = y;
		this.sigma = sigma;
		this.theta = theta;
		this.value = value;
	}
	
	public Double getMagnitudeSumByThetaRange(Double thetaMin, Double delta){
		Double sum = 0D;
		
//		for(Gradient[] gradientRow : getGradients()){
//			for(Gradient gradient : gradientRow){
//				if(gradient != null && gradient.getTheta() > thetaMin && gradient.getTheta() <= thetaMin + delta){
//					sum += gradient.getMagnitude();
//				}
//			}
//		}
		
		for(Histogram histogram : getDescriptor().getHistograms()){
			for(Entry<Double, Double> entry : histogram.entrySet()){
				if(entry.getKey() > thetaMin && entry.getKey() <= thetaMin + delta)	
					sum += entry.getValue();
			}
		}
		
		return normalizeValue(sum);
	}
	
	public Double normalizeValue(Double value){
		// Normalize value with small constant
		return value / Math.sqrt(Math.pow(value, 2D) + Math.pow(0.8, 2D));
	}

	public Double getX() {
		return x;
	}

	public void setX(Double x) {
		this.x = x;
	}

	public Double getY() {
		return y;
	}

	public void setY(Double y) {
		this.y = y;
	}

	public Double getSigma() {
		return sigma;
	}

	public void setSigma(Double sigma) {
		this.sigma = sigma;
	}

	public Double getTheta() {
		return theta;
	}

	public void setTheta(Double theta) {
		this.theta = theta;
	}

	public Double getValue() {
		return value;
	}

	public void setValue(Double value) {
		this.value = value;
	}


	public IPixelNode getKeyPointNode() {
		return (IPixelNode) keyPointNode;
	}


	public void setKeyPointNode(IPixelNode keyPointNode) {
		this.keyPointNode = keyPointNode;
	}
	
	public void produceCircle(){
		
		if(keyPointNode.getAreaSquare().getImageArea() != null){
			
			Long radius = null;
			GraphicsContext gc = keyPointNode.getAreaSquare().getImageArea().gc;
	        gc.setStroke(Color.RED);
	        gc.setLineWidth(1);
	        
	        // Cercle de rayon sigma autour du point d'interet
	        
			// We use the calcul of the FWHM to evaluate the radius at 1/1000 from 0 on y.
			// y= σ * sqrt{2ln(1000)} + x_{center}
			// radius = σ * sqrt{2ln(1000)} + x_{center}
			// radius = σ * 3,716922188849838 + x_{center}
			radius = Math.round(sigma * 3.7167D);
	        
	        gc.strokeOval(x - radius, y - radius, radius * 2D, radius * 2D);
	        
	        gc.setStroke(Color.AQUA);
	        gc.strokeLine(x, y, (Math.cos(theta) * radius) + x, (Math.sin(theta) * radius) + y);
		}
		
	}


	public KeyPointDescriptor getDescriptor() {
		return descriptor;
	}


	public void setDescriptor(KeyPointDescriptor descriptor) {
		this.descriptor = descriptor;
	}


	public void setGradients(Gradient[][] gradients) {
		this.gradients = gradients;
	}


	public Gradient[][] getGradients() {
		return gradients;
	}


	public Gradient getGradient(int x, int y) {
		Gradient gradient = null;
		try{
			gradient = gradients[x][y];
		}catch(ArrayIndexOutOfBoundsException aioob){
			//System.err.println("Impossible de recuperer le gradient pour (x,y) = (" + x + "," + y + ")");
		}
		
		return gradient;
	}


	@Override
	public String toString() {
		return String.format("KeyPoint [kpType=%s, x=%s, y=%s, sigma=%s, theta=%s, value=%s, descriptor=%s]", kpType, x, y, sigma, theta, value, descriptor);
	}


}
