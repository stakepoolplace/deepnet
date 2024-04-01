package RN.strategy;

/**
 * @author Eric Marchand
 *
 */
public class TestStrategy extends AbstractStrategyDecorator implements IStrategy{

	public TestStrategy(Strategy strategy) {
		this.decoratedStrategy = strategy;
	}

	@Override
	public void beforeProcess() {
		System.out.println("TestStrategy beforeProcess");
		
	}

	@Override
	public void afterProcess() {
		System.out.println("TestStrategy afterProcess");
		
	}

}
