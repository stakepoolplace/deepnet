package RN.algoactivations;


/**
 * @author Eric Marchand
 *
 */
public class LeakyReLUPerformer implements IActivation{
	
	public static Double alpha = 0.01D;

	@Override
	public double perform(double... value) throws Exception {
		
		if(value[0] > 0D)
			return value[0];
		else
			return alpha * value[0];
		
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		if(value[0] > 0D)
			return 1.0D;
		else
			return alpha;
	}

}
