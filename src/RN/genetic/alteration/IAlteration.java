package RN.genetic.alteration;

/**
 * @author Eric Marchand
 *
 */
public interface IAlteration {

	void beforeProcess();

	void afterProcess();
	
	void apply();
	
}
