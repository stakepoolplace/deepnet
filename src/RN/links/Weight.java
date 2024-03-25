package RN.links;

/**
 * @author Eric Marchand
 *
 */
public class Weight {
	
	private double weight = 1.0D;

	
	public Weight(double weight){
		this.weight = weight;
	}
	
	public Weight(){
	}
	
	
	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	
	public void add(double value){
		weight += value;
	}
	
	public String toString(){
		return Double.valueOf(weight).toString();
	}
	

}
