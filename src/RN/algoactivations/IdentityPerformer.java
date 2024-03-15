package RN.algoactivations;

import RN.algoactivations.utils.BoundNumbers;

public class IdentityPerformer implements IActivation{

	@Override
	public double perform(double... value) throws Exception {
		return BoundNumbers.bound(value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		return 1;
	}

}
