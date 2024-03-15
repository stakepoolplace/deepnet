package RN.linkage;

public class SigmaWi {
	
	Double sigmaWi = null;

	public SigmaWi() {
		this.sigmaWi = 0D;
	}

	public Double value() {
		return sigmaWi;
	}
	
	public Double sum(Double value) {
		
		this.sigmaWi += value;
		
		return sigmaWi;
	}
	
	public Double multiply(Double value){
		
		this.sigmaWi *= value;
		
		return sigmaWi;
	}

	public void setSigmaWi(Double sigmaWi) {
		this.sigmaWi = sigmaWi;
	}

}
