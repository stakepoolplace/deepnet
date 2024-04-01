package RN.algoactivations;

import java.io.Serializable;

import RN.algoactivations.utils.BoundMath;


/**
 * @author Eric Marchand
 *
 */
public class CosinusPerformer implements Serializable, IActivation {

	@Override
	public double perform(double... value) throws Exception {
		return BoundMath.cos(value[0]);
		
	}

	@Override
	public double performDerivative(double... value) {
		return -BoundMath.sin(value[0]);
	}

}
