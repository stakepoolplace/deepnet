package RN.algoactivations;

import RN.algoactivations.utils.BoundMath;


public class SygmoidInversePerformer implements IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		return -1.0D / (1.0D + BoundMath.exp(-value[0]));
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		return  -BoundMath.exp(value[0]) / ( Math.pow(BoundMath.exp(value[0]) + 1.0D, 2D) );
	}

}
