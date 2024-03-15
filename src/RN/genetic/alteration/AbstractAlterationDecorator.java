package RN.genetic.alteration;



public abstract class AbstractAlterationDecorator extends Alteration implements IAlteration{



	protected Alteration decoratedAlteration = null;
	
	
	public abstract void beforeProcess();
	
	@Override
	public void apply(){
		
		beforeProcess();
		
		if(decoratedAlteration != null)
			decoratedAlteration.apply();
		
		process();
		
		afterProcess();
		
		geneticCodeAlteration();
		
	}
	
	public abstract void afterProcess();

	public abstract String geneticCodeAlteration();
	
}
