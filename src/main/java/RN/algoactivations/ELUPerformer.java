package RN.algoactivations;

import java.io.Serializable;

import RN.algoactivations.utils.BoundMath;

/**
 * @author Eric Marchand
 *
 */
public class ELUPerformer implements Serializable, IActivation{
	
	public static Double alpha = 0.01D;

	@Override
	public double perform(double... value) throws Exception {
		
		if(value[0] > 0D)
			return value[0];
		else
			return alpha * (BoundMath.exp(value[0]) - 1.0D);
		
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		if(value[0] > 0D)
			return 1.0D;
		else
			return alpha * BoundMath.exp(value[0]);
	}

}
