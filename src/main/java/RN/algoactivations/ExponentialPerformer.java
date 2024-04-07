package RN.algoactivations;

import java.io.Serializable;

import RN.algoactivations.utils.BoundMath;


/**
 * @author Eric Marchand
 *
 */
public class ExponentialPerformer implements Serializable, IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		
		return BoundMath.exp(value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		
        // La dérivée de l'exponentielle est l'exponentielle de la valeur elle-même
        return BoundMath.exp(value[0]);
	}

}
