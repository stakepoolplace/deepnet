package RN.algoactivations;

import java.io.Serializable;

import RN.algoactivations.utils.BoundMath;


/**
 * @author Eric Marchand
 *
 */
public class SoftPlusPerformer implements Serializable, IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		
		return  BoundMath.log(1D + value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		double expx = BoundMath.exp(value[0]);
		
		return  expx / ( 1.0D + expx );
	}

}
