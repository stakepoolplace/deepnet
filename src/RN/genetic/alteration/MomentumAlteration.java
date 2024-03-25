package RN.genetic.alteration;

import RN.genetic.Genetic;
import RN.utils.StatUtils;

/**
 * @author Eric Marchand
 *
 */
public class MomentumAlteration extends AbstractAlterationDecorator implements IAlteration{

	private String geneticCodeAlteration = "";
	
	public MomentumAlteration(Alteration alteration) {
		this.decoratedAlteration = alteration;
	}

	public MomentumAlteration() {
	}

	@Override
	public void beforeProcess() {
		
	}

	@Override
	public void process() {
		System.out.print("momentum was : " +  trainer.getAlphaDeltaWeight());
		trainer.setAlphaDeltaWeight(Math.random());
		System.out.println(" set to  : " + trainer.getAlphaDeltaWeight());
		geneticCodeAlteration = "m%(" + StatUtils.format(trainer.getAlphaDeltaWeight(), "#.#") + ")" + Genetic.CODE_SEPARATOR;
	}

	@Override
	public void afterProcess() {
		
	}
	
	@Override
	public String geneticCodeAlteration(){
		
		network.setName(network.getName() + geneticCodeAlteration);
		
		return geneticCodeAlteration;
		
	}

}
