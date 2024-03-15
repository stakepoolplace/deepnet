package RN.algoactivations;

public class SygmoidPerformer2 implements IActivation {

	@Override
	public double perform(double... value) throws Exception {
		
		//return (2.0D / (1.0D + BoundMath.exp(-5*value))) - 1D;
		return (2.0D / (1.0D + Math.exp(-value[0]))) - 1D;
	}

	@Override
	public double performDerivative(double... value) throws Exception {
		
		//return  (10 * BoundMath.exp( 5 * value))  /  (Math.pow( 1.0D + BoundMath.exp(5 * value) , 2D)) ;
		return  (2D * Math.exp(value[0]))  /  (Math.pow( 1.0D + Math.exp(value[0]) , 2D)) ;
	}

}
