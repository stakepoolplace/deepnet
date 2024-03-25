package RN.utils;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * @author Eric Marchand
 *
 */
public class StatUtils {
	

	public static int randomize(int max){
		return (int) Math.floor(Math.random() * max + 0.5d);
	}
	
	public static double round(double value, int places) {
	    
		if (places < 0) throw new IllegalArgumentException();

	    long factor = (long) Math.pow(10, places);
	    value = value * factor;
	    long tmp = Math.round(value);
	    
	    return (double) tmp / factor;
	}
	
	public static String format(double value, String format){
		DecimalFormat df = new DecimalFormat(format);
		return df.format(value);
	}
	
	public static Double initValue(double min, double max) {
		Random random = new Random();
		return random.nextDouble() * (max - min) + min;
	}
	
	public static Double nextDouble() {
		Random random = new Random();
		return random.nextDouble();
	}

	public static double absoluteSum(Double[][] matrix) {
		Double sum = 0D;
		for(int i=0;i<matrix.length;i++){
			for(int j=0;j<matrix[i].length;j++){
				sum += Math.abs(matrix[i][j]);
			}
		}
		return sum;
	}
	
	
}
