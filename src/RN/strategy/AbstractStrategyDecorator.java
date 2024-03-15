package RN.strategy;



public abstract class AbstractStrategyDecorator extends Strategy implements IStrategy{



	protected Strategy decoratedStrategy = null;
	
	
	public abstract void beforeProcess();
	
	@Override
	public void apply(){
		
		beforeProcess();
		
		if(decoratedStrategy != null)
			decoratedStrategy.apply();
		
		process();
		
		afterProcess();
		
	}
	
	public abstract void afterProcess();
	
}
