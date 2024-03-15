package RN.algoactivations;

import RN.algoactivations.utils.BoundMath;


public class TanHPerformer implements IActivation {

	
	@Override
	public double perform(double... value) throws Exception {
		return 1.7159D * BoundMath.tanh(value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {

		double coshx = BoundMath.cosh(value[0]);
		double denom = BoundMath.cosh(2*value[0]) + 1;
		return (1.7159D * 4D * coshx * coshx) / (denom * denom);
	}

}
