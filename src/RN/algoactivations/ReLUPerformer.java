package RN.algoactivations;


/**
 * @author Eric Marchand
 *
 */
public class ReLUPerformer implements IActivation{

	@Override
	public double perform(double... value) throws Exception {
		
		return Math.max(0D, value[0]);
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		if(value[0] > 0D)
			return 1.0D;
		else
			return 0.0D;
	}

}
