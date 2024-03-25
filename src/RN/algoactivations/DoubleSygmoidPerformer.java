package RN.algoactivations;



/**
 * @author Eric Marchand
 *
 */
public class DoubleSygmoidPerformer implements IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		// y = sgn(x - d) ( 1 - exp(-(x-d / s)^2))
		
		return  0;//1.0D / (1.0D + BoundMath.exp(-value));
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		return 0;  //(BoundMath.exp(value)) / ( (BoundMath.exp(value) + 1.0D) * (BoundMath.exp(value) + 1.0D));
	}

}
