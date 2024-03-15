package RN.algoactivations;

import RN.algoactivations.utils.BoundMath;



public class CosinusPerformer implements IActivation {

	@Override
	public double perform(double... value) throws Exception {
		return BoundMath.cos(value[0]);
		
	}

	@Override
	public double performDerivative(double... value) {
		return -BoundMath.sin(value[0]);
	}

}
