package RN.dataset;

import java.util.Arrays;
import java.util.List;

public class InputData {

	public List<Double> input;
	public List<Double> ideal;
	
	public InputData(Double[] input, Double[] ideal){
		
		if(input != null)
			this.input = Arrays.asList(input);
		
		if(ideal != null)
			this.ideal = Arrays.asList(ideal);
		
	}
	
	

	public InputData(List<Double> input, List<Double> ideal){
		this.input = input;
		this.ideal = ideal;
	}
	
	public Double getInput(int ind) {
		return input.get(ind);
	}



	public List<Double> getInput() {
		return input;
	}



	public void setInput(List<Double> input) {
		this.input = input;
	}



	public List<Double> getIdeal() {
		return ideal;
	}

	public Double getIdeal(int ind) {
		return ideal.get(ind);
	}

	public void setIdeal(List<Double> ideal) {
		this.ideal = ideal;
	}

	public float[] getInputArray() {
		
		float[] res = new float[input.size()];
		int idx = 0;
		for(Double in : input){
			res[idx++] = in.floatValue();
		}
		
		
		return res;
	}


	public float[] getIdealArray() {
		float[] res = new float[ideal.size()];
		int idx = 0;
		for(Double in : ideal){
			res[idx++] = in.floatValue();
		}
		
		return res;
	}

	

	
	
	
}
