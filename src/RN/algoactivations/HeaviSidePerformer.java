package RN.algoactivations;


/**
 * 
 * Remarque : L'application de descente du gradient ne fonctionne pas
 * car Heaviside n'est pas dérivable.
 * 
 * @author Eric
 *
 */
public class HeaviSidePerformer implements IActivation {

	@Override
	public double perform(double... value) throws Exception {

		if(value[0] > 0D)
			return 1.0D;
		else
			return 0.0D;
		
	}

	@Override
	public double performDerivative(double... value) {
		return 1.0D;
	}

}
