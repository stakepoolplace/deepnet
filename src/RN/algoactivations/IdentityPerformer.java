package RN.algoactivations;

import java.io.Serializable;

import RN.algoactivations.utils.BoundNumbers;

/**
 * @author Eric Marchand
 *
 */
public class IdentityPerformer implements Serializable, IActivation{

	@Override
	public double perform(double... value) throws Exception {
		return BoundNumbers.bound(value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		return 1;
	}

}
