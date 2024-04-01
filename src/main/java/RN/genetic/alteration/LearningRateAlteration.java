package RN.genetic.alteration;

import RN.genetic.Genetic;
import RN.utils.StatUtils;

/**
 * @author Eric Marchand
 *
 */
public class LearningRateAlteration extends AbstractAlterationDecorator implements IAlteration{

	private String geneticCodeAlteration = "";
	
	public LearningRateAlteration(Alteration alteration) {
		this.decoratedAlteration = alteration;
	}

	public LearningRateAlteration() {
	}

	@Override
	public void beforeProcess() {
		
	}

	@Override
	public void process() {
		System.out.print("learning rate was : " + trainer.getLearningRate() );
		trainer.setLearningRate(Math.random());
		System.out.println(" set to  : " + trainer.getLearningRate());
		
		geneticCodeAlteration = "l%(" + StatUtils.format(trainer.getLearningRate(), "#.#") + ")" + Genetic.CODE_SEPARATOR;
		
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
