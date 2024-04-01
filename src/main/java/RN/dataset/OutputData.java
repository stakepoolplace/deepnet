package RN.dataset;

/**
 * @author Eric Marchand
 *
 */
public class OutputData {

	public Double[] output;
	
	public OutputData(Double[] output){
		this.output = output;
	}

	public Double[] getOutput() {
		return output;
	}
	
	public Double getOutput(int ind) {
		return Double.valueOf(output[ind]);
	}
	
	public void setOutput(Double[] output) {
		this.output = output;
	}
	
	public String toString(){
		String result = "";
		int idx = 0;
		for(Double val : output){
			result += "[" + idx++ + "] " + val + "    ";
		}
		return result;
	}

	
	
}
