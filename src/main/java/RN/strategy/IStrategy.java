package RN.strategy;

/**
 * @author Eric Marchand
 *
 */
public interface IStrategy {

	void beforeProcess();

	void afterProcess();
	
	void apply();
	
}
