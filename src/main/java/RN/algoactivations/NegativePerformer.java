package RN.algoactivations;

import java.io.Serializable;

/**
 * @author Eric Marchand
 *
 */
public class NegativePerformer implements Serializable, IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		return -value[0];
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		return  -1;
	}

}
