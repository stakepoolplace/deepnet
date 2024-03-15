package RN.algoactivations;

public class NegativePerformer implements IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		return -value[0];
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		return  -1;
	}

}
