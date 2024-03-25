package RN.utils;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * @author Eric Marchand
 *
 */
public class MathUtils {

	public MathUtils() {
		// TODO Auto-generated constructor stub
	}
	
	
	public static int odd(int value){
		return value % 2 == 0 ? 0 : 1;
	}
	
	public static int even(int value){
		return value % 2 == 0 ? 1 : 0;
	}
	
	public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    BigDecimal bd = new BigDecimal(value);
	    bd = bd.setScale(places, RoundingMode.HALF_UP);
	    return bd.doubleValue();
	}

}
