package RN.dataset;

/**
 * @author Eric Marchand
 *
 */
public class Coordinate {
	
	// x the abscissa coordinate with origin 0 of the image
	private Double x = null;
	
	// y the ordinate coordinate with origin 0 of the image
	private Double y = null;
	
	// cartesian center x
	private Double x0 = null;
	
	// cartesian center y
	private Double y0 = null;
	
	// angle in polar system
	private Double theta = null;
	
	// eccentricity, distance from the center image in polar system
	private Double r = null;
	
	// eccentricity, distance from the center image in log-polar system
	private Double p = null;
	
	// base for the log-polar system
	private Double base = null;
	
	
	public Coordinate() {
	}
	
	public Coordinate(double x, double y) {
		this.x = x;
		this.y = y;
	}
	
	public Coordinate(ECoordinateSystem system, double a, double b) {
		
		if(system == ECoordinateSystem.LINEAR){
			this.x = a;
			this.y = b;
		}
		
		if(system == ECoordinateSystem.POLAR){
			this.r = a;
			this.theta = b;
		}
		
		if(system == ECoordinateSystem.LOG_POLAR){
			this.p = a;
			this.theta = b;
		}
	}
	
	
	
	public void linearToPolarSystem(){
		
		double x1 = this.x - x0;
		double y1 = this.y - y0;
		
		this.r = Math.sqrt(x1*x1 + y1*y1);
		
		
		double theta1 = Math.atan2(y1, x1);
		
		while(theta1 < 0D){
			theta1 += Math.PI * 2D;
		}
		
		this.theta = theta1;
	}
	
	public void linearToLogPolarSystem(){
		
		linearToPolarSystem();
		
		if(this.base == null){
			this.p = Math.log10(this.r);
		}else{
			this.p = Math.log10(this.r) / Math.log10(this.base);
		}
		
	}
	
	public void polarToLinearSystem(){
		
		this.x = this.r * Math.cos(this.theta) + this.x0;
		this.y = this.r * Math.sin(this.theta) + this.y0;
		
	}
	
	public void logPolarToLinearSystem(){
		
		this.x = Math.pow(this.base, this.p) * Math.cos(this.theta) + this.x0;
		this.y = Math.pow(this.base, this.p) * Math.sin(this.theta) + this.y0;
		
	}
	
	public void rotateSystem(double angle){
		
		this.x = (int) Math.round((x-x0) * Math.cos(angle) - (y-y0) * Math.sin(angle)) + x0 ;
		this.y = (int) Math.round((x-x0) * Math.sin(angle) + (y-y0) * Math.cos(angle)) + y0 ;
		
		if(this.theta != null){
			linearToPolarSystem();
		}
	}
	
	private int log2(int value){
		return (int) (Math.log(value) / Math.log(2) + 1e-10);
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

	public Double getTheta() {
		return theta;
	}

	public void setTheta(Double theta) {
		this.theta = theta;
	}

	public Double getR() {
		return r;
	}

	public void setR(Double r) {
		this.r = r;
	}

	public Double getP() {
		return p;
	}

	public void setP(Double p) {
		this.p = p;
	}

	public Double getBase() {
		return base;
	}

	public void setBase(Double base) {
		this.base = base;
	}

	public Double getX0() {
		return x0;
	}

	public void setX0(Double x0) {
		this.x0 = x0;
	}

	public Double getY0() {
		return y0;
	}

	public void setY0(Double y0) {
		this.y0 = y0;
	}
	
	

}
