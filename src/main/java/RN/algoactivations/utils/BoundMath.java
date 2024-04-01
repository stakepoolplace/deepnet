package RN.algoactivations.utils;

/**
 * Avoid Math.Nan
 * @author Eric Marchand
 */
public final class BoundMath {

	/**
	 * Calculate the cos.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double cos(final double a) {
		return BoundNumbers.bound(Math.cos(a));
	}

	/**
	 * Calculate the exp.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double exp(final double a) {
		return BoundNumbers.bound(Math.exp(a));
	}

	/**
	 * Calculate the log.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double log(final double a) {
		return BoundNumbers.bound(Math.log(a));
	}

	/**
	 * Calculate the power of a number.
	 * 
	 * @param a
	 *            The base.
	 * @param b
	 *            The exponent.
	 * @return The result of the function.
	 */
	public static double pow(final double a, final double b) {
		return BoundNumbers.bound(Math.pow(a, b));
	}

	/**
	 * Calculate the sin.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double sin(final double a) {
		return BoundNumbers.bound(Math.sin(a));
	}

	/**
	 * Calculate the square root.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double sqrt(final double a) {
		return Math.sqrt(a);
	}

	/**
	 * Private constructor.
	 */
	private BoundMath() {

	}

	/**
	 * Calculate TANH, within bounds.
	 * @param d The value to calculate for.
	 * @return The result.
	 */
	public static double tanh(final double d) {
		return BoundNumbers.bound(Math.tanh(d));
	}
	
	/**
	 * Calculate COSH, within bounds.
	 * @param d The value to calculate for.
	 * @return The result.
	 */
	public static double cosh(final double d) {
		return BoundNumbers.bound(Math.cosh(d));
	}
	
	
}
