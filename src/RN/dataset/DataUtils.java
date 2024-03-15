package RN.dataset;

import java.util.ArrayDeque;
import java.util.Deque;

public class DataUtils {

	private static Integer latestPeriodsCount = 26;
	private static DataUtils instance = null;

	
	public static DataUtils getInstance(){
		
		if(instance == null){
			instance = new DataUtils();
		}
		return instance;
	}

	public double getExponentialMean(Deque<Double> latestInputValues, int N) {
		
		
		double alpha = 2.0d / (N + 1.0d);
		double result = 0.0d;
		double n = 1.0d;

		for (double value : latestInputValues) {
			result += Math.pow(1.0d - alpha, n) * value;
			n++;
		}

		result *= alpha;

//		System.out.println("Si x(t) varie en moins de " + (-(2.0d * Math.PI) / Math.log(1d - alpha))
//				+ " échantillons, la fluctuation se retrouve dans la moyenne exp. mais est d'autant plus affaiblie qu'elle est rapide.");

		return result;

	}



	public double MACD(double lastMACD, double MACD, double lastSignal, double signal, double lastHistogram, double histogram, double stockPrice) {

		return histogram;
		
		// the MACD line crosses the signal line
//		if (histogram == 0)
//			return 1.0D;
//		 if(lastHistogram <= 0 && histogram > 0)
//			 return 1.0D;
//		 if(lastHistogram > 0 && histogram <= 0)
//			 return 1.0D;
//
//		// the MACD line crosses zero
//		 if(lastMACD <= 0 && MACD > 0)
//			 return 1.0D;
//		 if(lastMACD > 0 && MACD <= 0)
//			 return 1.0D;
//		 if(MACD == 0)
//			 return 1.0D;
//		 
//		 return 0.0D;
			 
		// there is a divergence between the MACD line and the price of the
		// stock or between the histogram and the price of the stock
		// higher highs (lower lows) on the price graph but not on the blue line, or higher highs (lower lows) on the price graph but not on the bar graph
		// Sign (relative price extremumfinal – relative price extremuminitial) ≠ Sign (relative MACD extremumfinal – MACD extremuminitial)
		 // TODO
	}
	
	public double RSI(double stockPrice) {


//		Float lastestValueSeries1 = latestSeries1.peekFirst();
//		Float lastestValueSeries2 = latestSeries1.peekFirst();
//		Float lastestValueSeries3 = latestSeries1.peekFirst();
//		
//		double closeNow = stockPrice;
//		double closePrevious = latestSeries1.peekFirst();
//		double U = closeNow - closePrevious;
//		
//		
//		double lastMACD = getExponentialMean(latestSeries1, null, 12) - getExponentialMean(latestSeries2, null, 26);
//		double lastSignal = getExponentialMean(latestSeries3, null, 9);
//		double lastHistogram = lastMACD - lastSignal;
//		
//		double MACD = getExponentialMean(latestSeries1, stockPrice, 12) - getExponentialMean(latestSeries2, stockPrice, 26);
//		double signal = getExponentialMean(latestSeries3, MACD, 9);
//		double histogram = MACD - signal;
//
//		// the MACD line crosses the signal line
//		if (histogram == 0)
//			return 1.0D;
//		 if(lastHistogram <= 0 && histogram > 0)
//			 return 1.0D;
//		 if(lastHistogram > 0 && histogram <= 0)
//			 return 1.0D;
//
//		// the MACD line crosses zero
//		 if(lastMACD <= 0 && MACD > 0)
//			 return 1.0D;
//		 if(lastMACD > 0 && MACD <= 0)
//			 return 1.0D;
//		 if(MACD == 0)
//			 return 1.0D;
		 
		 return 0.0D;
			 
		// there is a divergence between the MACD line and the price of the
		// stock or between the histogram and the price of the stock
		// higher highs (lower lows) on the price graph but not on the blue line, or higher highs (lower lows) on the price graph but not on the bar graph
		// Sign (relative price extremumfinal – relative price extremuminitial) ≠ Sign (relative MACD extremumfinal – MACD extremuminitial)
		 // TODO
	}

}
