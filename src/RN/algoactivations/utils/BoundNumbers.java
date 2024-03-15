package RN.algoactivations.utils;

/**
 * A simple class that prevents numbers from getting either too big or too
 * small.
 */
public final class BoundNumbers {

	/**
	 * Too small of a number.
	 */
	public static final double TOO_SMALL = -1.0E20;

	/**
	 * Too big of a number.
	 */
	public static final double TOO_BIG = 1.0E20;

	/**
	 * Bound the number so that it does not become too big or too small.
	 * 
	 * @param d
	 *            The number to check.
	 * @return The new number. Only changed if it was too big or too small.
	 */
	public static double bound(final double d) {
		if (d < BoundNumbers.TOO_SMALL) {
			return BoundNumbers.TOO_SMALL;
		} else if (d > BoundNumbers.TOO_BIG) {
			return BoundNumbers.TOO_BIG;
		} else {
			return d;
		}
	}

	/**
	 * Private constructor.
	 */
	private BoundNumbers() {

	}
}
