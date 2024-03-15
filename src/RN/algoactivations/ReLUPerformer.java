package RN.algoactivations;


public class ReLUPerformer implements IActivation{

	@Override
	public double perform(double... value) throws Exception {
		
		return Math.max(0D, value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		return 1D;
	}

}
