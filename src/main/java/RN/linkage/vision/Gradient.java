package RN.linkage.vision;

import RN.nodes.IPixelNode;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * @author Eric Marchand
 *
 */
public class Gradient {
	
	private IPixelNode pixNode = null; 
	
	private Double theta = null;
	
	private Double magnitude = null;
	
	private Double distanceToKeyPoint = null;
	
	private static Double maxMagnitude = 0D;

	public Gradient(IPixelNode pixNode, Double theta, Double value, Double distanceToKeyPoint) {
		this.pixNode = pixNode;
		this.theta = theta;
		this.magnitude = value;
		this.distanceToKeyPoint = distanceToKeyPoint;
		maxMagnitude = magnitude > maxMagnitude ? magnitude : maxMagnitude;
	}

	public Double getTheta() {
		return theta;
	}

	public void setTheta(Double theta) {
		this.theta = theta;
	}

	public Double getMagnitude() {
		return magnitude;
	}

	public Double getDistanceToKeyPoint() {
		return distanceToKeyPoint;
	}

	public void thresholdAtMax(double r) {
		magnitude = Math.min(magnitude, r * maxMagnitude);
	}

	public void thresholdAt(double threshold) {
		magnitude = Math.min(magnitude, threshold);
	}
	
	public void produceGradient(double magnitudeFactor, Color color){
		
		if(pixNode.getAreaSquare().getImageArea() != null){
			GraphicsContext gc = pixNode.getAreaSquare().getImageArea().gc;
	        gc.setLineWidth(1);
	        if(color == null)
	        	gc.setStroke(Color.AQUA);
	        else
	        	gc.setStroke(color);
	        
	        //gc.strokeOval(pixNode.getX(), pixNode.getY(), 0.1, 0.1);
	        gc.strokeLine(pixNode.getX(), pixNode.getY(), (Math.cos(theta) * magnitude * magnitudeFactor) + pixNode.getX(), (Math.sin(theta) * magnitude * magnitudeFactor) + pixNode.getY());
		}
		
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((magnitude == null) ? 0 : magnitude.hashCode());
		result = prime * result + ((theta == null) ? 0 : theta.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Gradient other = (Gradient) obj;
		if (magnitude == null) {
			if (other.magnitude != null)
				return false;
		} else if (!magnitude.equals(other.magnitude))
			return false;
		if (theta == null) {
			if (other.theta != null)
				return false;
		} else if (!theta.equals(other.theta))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return String.format("Gradient [pixNode=%s, theta=%s, magnitude=%s, distanceToKeyPoint=%s]", pixNode, theta, magnitude, distanceToKeyPoint);
	}




}
