package RN.dataset.inputsamples;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;

import RN.DataSeries;
import RN.EBoolean;
import RN.ITester;
import RN.NetworkElement;
import RN.ViewerFX;
import RN.dataset.Coordinate;
import RN.dataset.DataUtils;
import RN.utils.StreamUtils;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;

/**
 * 
 * @author Eric Marchand
 * 
 */
public class InputSample extends NetworkElement{

	private static InputSample instance = null;

	private String name;
	private ESamples sample;
	private int fileSheetIdx;
	public static double[] Ikeda;

	public InputSample() {
		// this.name = name;
		// this.sample = sample;
		initIkeda();
	}

	public InputSample(String name, ESamples sample) {
		this.name = name;
		this.sample = sample;
		initIkeda();
		getInstance();
	}

	public InputSample(String name, ESamples sample, int idx) {
		this.name = name;
		this.sample = sample;
		this.fileSheetIdx = idx;
		initIkeda();
		getInstance();
	}

	public static InputSample getInstance() {
		if (instance == null) {
			instance = new InputSample();
		}
		return instance;
	}

	@Override
	public String toString() {
		return this.name;
	}

	private Deque<Double> latestSeries1;
	private Deque<Double> latestSeries2;
	private Deque<Double> latestSeries3;
	

	public Double compute(ESamples function, Double... paramDouble) {
		double d1 = 0;

		if (function == null)
			return paramDouble[0];
		
		
		if (function == ESamples.LOG_GABOR) {

			if (paramDouble.length != 9)
				throw new RuntimeException("Missing Gabor parameter's");

			double n = paramDouble[0];
			
			double x = paramDouble[1];
			double y = paramDouble[2];
			
			double x0 = paramDouble[3];
			double y0 = paramDouble[4];
			
			//Number of scales of the multiresolution scheme
			double n_s = paramDouble[5];
			double s = paramDouble[6];
			
			//Number of orientations (between 3 to 20) 8 is a typical value
			double t = paramDouble[7];
			double n_t = paramDouble[8];

			// (p,theta) are the log-polar coordinate (in log2 scale)
			Coordinate coord = new Coordinate(x,y);
			coord.setX0(x0);
			coord.setY0(y0);
			coord.setBase(2D);
			coord.linearToLogPolarSystem();
			
			double p = coord.getP();
			double theta = coord.getTheta();

			// centre du filtre
			double r_s = Math.log10(n) / Math.log10(2) - s;
			double theta_st = s % 2D == 0D ? (Math.PI / n_t) * (t + 0.5D) : (Math.PI / n_t) * t;

			// variance selon r et theta
			double sigma_theta = 0.996D * ( Math.PI / (n_t * Math.sqrt(2D)));
			double sigma_r = 0.996D * Math.sqrt(2D / 3D);
			
			d1 = Math.exp(- Math.pow(p - r_s, 2D) / (2D * sigma_r * sigma_r));
			
			d1 *=  Math.exp(- Math.pow(theta - theta_st, 2D) / (2D * sigma_theta * sigma_theta));
			
		}else if (function == ESamples.SAMPLING) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing sampling parameter's");

			double x = paramDouble[0];
			double y = paramDouble[1];
			double sampling = paramDouble[2];

			d1 = (x % sampling == 0  ? 1D : 0D);
			d1 *= (y % sampling == 0  ? 1D : 0D);
			
		}else if (function == ESamples.GAUSSIAN) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			d1 = Gxy( x,  x1,  ox,  y,  y1,  oy);
		
			
		}else if (function == ESamples.GAUSSIAN_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			double k = (paramDouble.length == 9 && paramDouble[8] != null ? paramDouble[8] : 1);
			
			if(k != 1){
				d1 =  ( 1D / (2D * Math.PI * k * Math.pow(ox,2D)) ) * Math.exp( -(Math.pow(x - x1, 2D) + Math.pow(y - y1, 2D))  /   (2D * k * ox * oy)  );
			}else{
				d1 =  Gxy_DE_MARR(x, x1, ox, y, y1, oy);
			}
			
		}else if (function == ESamples.G_Dx_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			d1 = G_Dx_DE_MARR(x, x1, ox, y, y1, oy);
		
			
		}else if (function == ESamples.G_Dy_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			d1 = G_Dy_DE_MARR(x, x1, ox, y, y1, oy);
		
			
		}else if (function == ESamples.G_Dxy_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			Double k = paramDouble.length == 9 ? paramDouble[8] : null;
			
			if(k != null)
				d1 = G_Dxy_DE_MARR(x, x1, ox, y, y1, oy, k);
			else
				d1 = G_Dxy_DE_MARR(x, x1, ox, y, y1, oy);
			
		}else if (function == ESamples.G_Dxx_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			Double k = paramDouble.length == 9 ? paramDouble[8] : null;
			
			if(k != null)
				d1 = G_Dxx_DE_MARR(x, x1, ox, y, y1, oy, k);
			else
				d1 = G_Dxx_DE_MARR(x, x1, ox, y, y1, oy);
			
		}else  if (function == ESamples.G_Dyy_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];

			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			Double k = paramDouble.length == 9 ? paramDouble[8] : null;
			
			if(k != null)
				d1 = G_Dyy_DE_MARR(x, x1, ox, y, y1, oy, k);
			else
				d1 = G_Dyy_DE_MARR(x, x1, ox, y, y1, oy);
			
		}else if (function == ESamples.G_D2xy_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 = G_D2xy_DE_MARR(x, x1, ox, y, y1, oy);
			
		}else if (function == ESamples.G_D2xyTheta_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];
			
			double theta = paramDouble[8];
			
			Double k = paramDouble.length == 10 ? paramDouble[9] : null;
			
			if(k != null)
				d1 = G_D2xyTheta_DE_MARR(x, x1, ox, y, y1, oy, theta, k);
			else
				d1 = G_D2xyTheta_DE_MARR(x, x1, ox, y, y1, oy, theta);

			
		}else if (function == ESamples.G_D3xy_DE_MARR) {

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 = G_Dxxx_DE_MARR(x, x1, ox, y, y1, oy) + G_Dyyy_DE_MARR(x, x1, ox, y, y1, oy);
			
		}else if (function == ESamples.G_Dxxy_DE_MARR) {
			
//			Le traitement repose sur cinq paramètres :
//				N représente la taille du masque (matrice carrée) implantant le filtre LOG. N est impair.
//				σ permet d'ajuster la taille du chapeau mexicain.
//				∆x et ∆y sont les pas d'échantillonnage utilisés pour discrétiser h''(x,y). Généralement ∆x = ∆ y
//				S est le seuil qui permet de sélectionner les contours les plus marqués.
//				Il est à noter que le choix des paramètres N, σ et ∆x ne doit pas se faire de façon indépendante. En effet, le masque, même de taille réduite, doit ressembler à un chapeau mexicain. Le problème ici est le même que celui que l'on rencontre lors de l'échantillonnage d'une fonction gaussienne. Le nombre de points N à considérer doit être tel que l'étendue occupe l'intervalle [-3σ , 3σ].
//				En fonction du pas d'échantillonnage, l'étendue spatiale vaut : (N-1) ∆x  .
//				Cette étendue peut aussi s'écrire en fonction de σ : (N-1) ∆x = kσ  avec k entier.
//				En prenant par exemple  ∆x = 1 , il s'agit de choisir N et σ de sorte que l'étendue du chapeau mexicain soit pertinente. Pour le chapeau mexicain, la valeur de k doit être au moins de 4.

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 = G_Dxxy_DE_MARR(x, x1, ox, y, y1, oy);
			
		} else if (function == ESamples.G_Dxyy_DE_MARR) {
			

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			
			d1 = G_Dxyy_DE_MARR(x, x1, ox, y, y1, oy);
			
		} else if (function == ESamples.G_Dyyy_DE_MARR) {
			

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 = G_Dyyy_DE_MARR( x, x1, ox, y, y1, oy);
			
		} else if (function == ESamples.G_Dxxx_DE_MARR) {
			

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 = G_Dxxx_DE_MARR(x, x1, ox, y, y1, oy);
			
			
			
		} else if (function == ESamples.G_COURBURE_DE_MARR) {
			

			if (paramDouble.length == 1)
				throw new RuntimeException("Missing gaussian parameter's");

			double widthPx = paramDouble[0];
			double heightPx = paramDouble[1];
			double x = paramDouble[2];
			double y = paramDouble[3];


			// centre
			double x1 = paramDouble[4];
			double y1 = paramDouble[5];

			// ecartement selon x et y
			double ox = paramDouble[6];
			double oy = paramDouble[7];

			d1 =  Math.abs(G_Dxx_DE_MARR(x, x1, ox, y, y1, oy) * G_Dyy_DE_MARR(x, x1, ox, y, y1, oy) - Math.pow(G_Dxy_DE_MARR(x, x1, ox, y, y1, oy), 2D));
			
			
		} else if (function == ESamples.COSINUS) {
			paramDouble[0] *= 16.0D;
			d1 = 0.5D + Math.cos(paramDouble[0] * 3.141592653589793D) / 2.0D;
			
		}else if(function == ESamples.SINUS){
			paramDouble[0] *= 4.0D;
			d1 = 0.5D + Math.sin(paramDouble[0] * 4.0D * 3.141592653589793D) / 2.0D;
			
		}else if(function == ESamples.COMPLEX){ 
			d1 = 0.5D + (Math.sin(paramDouble[0] * 3.0D * 3.141592653589793D) + Math.sin(paramDouble[0] * 7.0D * 3.141592653589793D)
					+ Math.sin(paramDouble[0] * 8.0D * 3.141592653589793D) + Math.sin(paramDouble[0] * 11.0D * 3.141592653589793D)) / 8.0D;
			
		}else if(function == ESamples.CHAOS){
			if (paramDouble[0] >= 2.0D) {
				d1 = Ikeda[201];
			} else {
				int i = (int) (paramDouble[0] * 100.0D);
				double d2 = 100.0D * paramDouble[0] - i;
				d1 = Ikeda[i] * (1.0D - d2) + Ikeda[(i + 1)] * d2;
			}
			
		}else if(function == ESamples.MEANEXP1){ 
			latestSeries1 = manageLatestSeries(latestSeries1, paramDouble[0], 5);
			d1 = DataUtils.getInstance().getExponentialMean(latestSeries1, 5);
			
		}else if(function == ESamples.MEANEXP2){ 
			latestSeries2 = manageLatestSeries(latestSeries2, paramDouble[0], 10);
			d1 = DataUtils.getInstance().getExponentialMean(latestSeries2, 10);
			
		}else if(function == ESamples.MEANEXP3){ 
			latestSeries3 = manageLatestSeries(latestSeries3, paramDouble[0], 15);
			d1 = DataUtils.getInstance().getExponentialMean(latestSeries3, 15);
			
		}else if(function == ESamples.MACD){ 
			latestSeries1 = manageLatestSeries(latestSeries1, null, 12);
			latestSeries2 = manageLatestSeries(latestSeries2, null, 26);
			latestSeries3 = manageLatestSeries(latestSeries3, null, 9);
			double lastMACD = DataUtils.getInstance().getExponentialMean(latestSeries1, 12) - DataUtils.getInstance().getExponentialMean(latestSeries2, 26);
			double lastSignal = DataUtils.getInstance().getExponentialMean(latestSeries3, 9);
			double lastHistogram = lastMACD - lastSignal;

			latestSeries1 = manageLatestSeries(latestSeries1, paramDouble[0], 12);
			latestSeries2 = manageLatestSeries(latestSeries2, paramDouble[0], 26);
			double MACD = DataUtils.getInstance().getExponentialMean(latestSeries1, 12) - DataUtils.getInstance().getExponentialMean(latestSeries2, 26);
			latestSeries3 = manageLatestSeries(latestSeries3, MACD, 9);
			double signal = DataUtils.getInstance().getExponentialMean(latestSeries3, 9);
			double histogram = MACD - signal;
			d1 = DataUtils.getInstance().MACD(lastMACD, MACD, lastSignal, signal, lastHistogram, histogram, paramDouble[0]);
			
		}else if(function == ESamples.RAND){
			d1 = Math.random();

		}else if(function == ESamples.TIMESERIE){ 
			// centrer reduit ?
		}else{ //IDENTITY
			d1 = paramDouble[0];
		}

		return d1;
	}
	
	
	private double Gxy(double x, double x1, double ox, double y, double y1, double oy) {
		
		return Math.exp((- Math.pow(x - x1, 2D) - Math.pow(y - y1, 2D))  /  (2D * ox * oy));
	}
	
	private double Gxy(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		return Math.exp((- Math.pow(x - x1, 2D) - Math.pow(y - y1, 2D))  /  (2D * k * ox * oy));
	}
	
	private double Gx(double x, double x1, double ox) {
		
		double result;
		double term1 = Math.pow(x - x1, 2D)  /   (2D * Math.pow( ox, 2D));
		
		result = Math.exp(-term1);
		
		return result;
	}
	
	private double Gy(double y, double y1, double oy) {
		
		double result;
		double term2 = Math.pow(y - y1, 2D)  /  (2D * Math.pow( oy, 2D));
		
		result = Math.exp(-term2);
		
		return result;
	}
	
	private double Gx_DE_MARR(double x, double x1, double ox) {
		
		double result;
		double term1 = ( 1D / (2D * Math.PI * Math.pow(ox,2D)) );
		
		result = term1 * Gx(x, x1, ox);
		
		return result;
	}
	
	
	private double Gy_DE_MARR(double y, double y1, double oy) {
		
		double result;
		double term1 = ( 1D / (2D * Math.PI * Math.pow(oy,2D))) ;
		
		result = term1 * Gy(y, y1, oy);
		
		return result;
	}
	
	private double Gxy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		double term1 = ( 1D / (2D * Math.PI * Math.pow(ox,2D)) );
		
		result = term1 * Gx(x, x1, ox) * Gy(y, y1, oy);
		
		return result;
	}
	
	private double G_Dx(double x, double x1, double ox) {
		
		double result;
		double term1 =  -(x - x1)  /  Math.pow( ox, 2);
		
		result = term1 * Gx_DE_MARR(x, x1, ox);
		
		return result;
	}
	
	private double G_Dy(double y, double y1, double oy) {
		
		double result;
		double term1 =  -(y - y1)  /  Math.pow( oy, 2);
		
		result = term1 * Gy_DE_MARR(y, y1, oy);
		
		return result;
	}
	
	private double G_Dxx(double x, double x1, double ox, double y, double y1, double oy) {
		
		double term1 =  (Math.pow(x - x1, 2D) - Math.pow( ox, 2D))  /  (2D * Math.PI * Math.pow( ox, 6D));
		
		return term1 * Gxy(x, x1, ox, y, y1, oy);
	}
	
	private double G_Dxx(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		double term1 =  (Math.pow(x - x1, 2D) - k * Math.pow( ox, 2D))  /  (2D * Math.PI * k * Math.pow( ox, 6D));
		
		return term1 * Gxy(x, x1, ox, y, y1, oy, k);
	}
	
	/**
	 * Dérivée Seconde partielle en x puis en y
	 * @param x
	 * @param x1
	 * @param ox
	 * @param y
	 * @param y1
	 * @param oy
	 * @return
	 */
	private double G_Dxy(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		result =  ((x-x1)*(y-y1)) / (2D * Math.PI * Math.pow(ox, 3D) * Math.pow(oy, 3D) ) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxy(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		double result;
		
		result =  ((x-x1)*(y-y1)) / (2D * Math.PI * k * Math.pow(ox, 3D) * Math.pow(oy, 3D) ) * Gxy(x, x1, ox, y, y1, oy, k);
		
		return result;
	}
	
	/**
	 * Dérivée Seconde de la gaussienne de marr en 2D
	 * @param x
	 * @param x1
	 * @param ox
	 * @param y
	 * @param y1
	 * @param oy
	 * @return
	 */
	private double G_D2xy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		//result =  G_Dxx(x, x1, ox) * G_Dyy(y, y1, oy);
		
		double term1 =  (-1D / (Math.PI * Math.pow(ox, 4D))) * (1D - ( Math.pow(x-x1, 2D) + Math.pow(y-y1, 2D)) / (2D * Math.pow(ox, 2D)))  ;
		
		result = term1 * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
//	private double G_D2xx_DE_MARR(double x, double x1, double ox) {
//		
//		double result;
//		
//		double term1 =  (Math.pow(x-x1, 2D) - Math.pow(ox, 2D)) / (  Math.PI * Math.pow(ox, 6D));
//		result = term1 * Gx(x, x1, ox);
//		
//		return result;
//	}
	
//	private double G_D2yy_DE_MARR( double y, double y1, double oy) {
//		
//		double result;
//		
//		double term1 =  (Math.pow(y-y1, 2D) - Math.pow(oy, 2D)) / (  Math.PI * Math.pow(oy, 6D));
//		result = term1 * Gy(y, y1, oy);		
//		
//		return result;
//	}
	
	private double G_D2xyTheta_DE_MARR(double x, double x1, double ox, double y, double y1, double oy, double theta) {
		
		double result;
		
		//theta = Math.atan2( G_Dy(y, y1, oy),  G_Dx(x, x1, ox));
		
		theta = theta * (Math.PI / 180D);
		
		result =  (Math.pow(Math.cos(theta), 2D) * G_Dxx(x, x1, ox, y, y1, oy)) + (2D * Math.cos(theta) * Math.sin(theta) * G_Dxy(x, x1, ox, y, y1, oy)) + (Math.pow(Math.sin(theta), 2D) * G_Dyy(x, x1, ox, y, y1, oy));
		
		
		return result;
	}
	
	private double G_D2xyTheta_DE_MARR(double x, double x1, double ox, double y, double y1, double oy, double theta, double k) {
		
		double result;
		
		theta = theta * (Math.PI / 180D);
		
		result =  (Math.pow(Math.cos(theta), 2D) * G_Dxx(x, x1, ox, y, y1, oy, k)) + (2D * Math.cos(theta) * Math.sin(theta) * G_Dxy(x, x1, ox, y, y1, oy, k)) + (Math.pow(Math.sin(theta), 2D) * G_Dyy(x, x1, ox, y, y1, oy, k));
		
		
		return result;
	}
	
	private double G_Dx_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( -(x - x1) / (2D * Math.PI * Math.pow(ox, 4D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( -(y - y1) / (2D * Math.PI * Math.pow(oy, 4D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		double result;
		
		
		result = ( ((x - x1) * (y - y1)) / (2D * Math.PI * k * Math.pow(ox, 4D) * Math.pow(oy, 2D))) * Gxy(x, x1, ox, y, y1, oy, k);
		
		return result;
	}
	
	private double G_Dxy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( ((x - x1) * (y - y1)) / (2D * Math.PI  * Math.pow(ox, 4D) * Math.pow(oy, 2D))) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxx_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		return ( (Math.pow(x - x1, 2D) - Math.pow(ox, 2D)) / (2D * Math.PI * Math.pow(ox, 6D) )) * Gxy(x, x1, ox, y, y1, oy);
		
	}
	
	private double G_Dxx_DE_MARR(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		return ( (Math.pow(x - x1, 2D) -  k * Math.pow(ox, 2D)) / (2D * Math.PI * k * Math.pow(ox, 6D) )) * Gxy(x, x1, ox, y, y1, oy, k);
		
	}
	
	private double G_Dyy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		return ( (Math.pow(y - y1, 2D) - Math.pow(oy, 2D)) / (2D * Math.PI * Math.pow(oy, 6D) )) * Gxy(x, x1, ox, y, y1, oy);
	}
	
	private double G_Dyy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		return ( (Math.pow(y - y1, 2D) - k * Math.pow(oy, 2D)) / (2D * Math.PI * k * Math.pow(oy, 6D) )) * Gxy(x, x1, ox, y, y1, oy, k);
	}
	
	private double G_Dyyy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( ( -Math.pow(y - y1, 3D) + 3D * (y-y1) * Math.pow(oy, 2D) ) / (2D * Math.PI * Math.pow(oy, 8D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxxx_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( ( -Math.pow(x - x1, 3D) + 3D * (x-x1) * Math.pow(ox, 2D) ) / (2D * Math.PI * Math.pow(ox, 8D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxxy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( ( (y-y1) * Math.pow(ox, 2D) - (y-y1) * Math.pow(x - x1, 2D)) / (2D * Math.PI * Math.pow(ox, 8D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dxyy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		
		
		result = ( ( (x-x1) * Math.pow(ox, 2D) - (x-x1) * Math.pow(y - y1, 2D)) / (2D * Math.PI * Math.pow(oy, 8D) )) * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
//	private double G_D3xy_DE_MARR(double x, double x1, double ox, double y, double y1, double oy) {
//		
//		double result;
//		
//		result = ((-Math.pow(x-x1, 3D) -Math.pow(y-y1, 3D) + ( ox * oy * (x-x1 + y-y1))) / (2D * Math.PI * Math.pow(ox, 8D))) * Gxy(x, x1, ox, y, y1, oy);
//		
//		return result;
//	}
	
	private double G_Dyy(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		double term1 =  (Math.pow(y-y1, 2D) - Math.pow( oy, 2D))  /  (2D * Math.PI * Math.pow( oy, 6D));
		
		result = term1 * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	private double G_Dyy(double x, double x1, double ox, double y, double y1, double oy, double k) {
		
		double result;
		double term1 =  (Math.pow(y-y1, 2D) - k * Math.pow( oy, 2D))  /  (2D * Math.PI * k * Math.pow( oy, 6D));
		
		result = term1 * Gxy(x, x1, ox, y, y1, oy, k);
		
		return result;
	}
	
	private double G_Dxxx(double x, double x1, double ox, double y, double y1, double oy) {
		
		double result;
		double term1 =  ( 3D * (x-x1) * Math.pow( ox, 2D) - Math.pow(x-x1, 3D) )  /  (2D * Math.PI * Math.pow( ox, 8D));
		
		result = term1 * Gxy(x, x1, ox, y, y1, oy);
		
		return result;
	}
	
	
	
	
	private double DerivatedGaussian(int n, double x, double x1, double sigmax){
		
		return Hnsigma(n, sigmax, x) * Gx(x, x1, sigmax);
	}
	
	private double Hnsigma(int n, double sigmax, double x){
		return (Math.pow(-1D, n) / (sigmax * Math.sqrt(2D)) * PolynomeHermite(n, x / (sigmax * Math.sqrt(2D))));
	}
	
	private Double PolynomeHermite(int n, double value){
		if(n==0){
			return 1D;
		}else if(n==1){
			return 2D * value;
		}else if(n==2){
			return 4D * Math.pow(value, 2D) - 2D;
		}else if(n==3){
			return 8D * Math.pow(value, 3D) - (12D * value);
		}else if(n==4){
			return 16D * Math.pow(value, 4D) - (48D * Math.pow(value, 2D) + 12D);
		}else if(n==5){
			return 32D * Math.pow(value, 5D) - (160D * Math.pow(value, 3D) + (120D * value));
		}else if(n==6){
			return 64D * Math.pow(value, 6D) - (480D * Math.pow(value, 4D) + (720D * Math.pow(value, 2D)) - 120D );
		}
		return null;
	}
	
	  

	private Deque<Double> manageLatestSeries(Deque<Double> latestInputValues, Double inputValue, int i) {

		int latestPeriodsCount = 26;
		if (latestInputValues == null)
			latestInputValues = new ArrayDeque<Double>(i);

		if (i > latestPeriodsCount)
			throw new RuntimeException("depassement de longueur, rallonger les historiques de donnees");

		if (inputValue == null)
			return latestInputValues;

		if (latestInputValues.isEmpty()) {
			for (int idx = 0; idx < latestPeriodsCount - 1; idx++)
				latestInputValues.addFirst(inputValue);
		}

		if (latestInputValues.size() > 0)
			latestInputValues.removeLast();

		latestInputValues.addFirst(inputValue);

		return latestInputValues;
	}

	public static void initIkeda() {
		double d1 = 0.0D;
		double d2 = 0.0D;
		Ikeda = new double[201];
		for (int i = 0; i <= 200; i++) {
			double d3 = 0.4D - 6.0D / (1.0D + d1 * d1 + d2 * d2);
			d1 = 1.0D + 0.7D * (d1 * Math.cos(d3) - d2 * Math.sin(d3));
			d2 = 0.7D * (d1 * Math.sin(d3) + d2 * Math.sin(d3));
			Ikeda[i] = (d1 - 0.2D);
		}
	}

	public static void setSamples(int inputCount, int outputCount, double delta, double delay) {

		DataSeries.getInstance().clearSeries();

		double d1 = (inputCount + outputCount - 2) * delta + delay;
		double d2;
		double d3;

		for (double j = 0; j < 500; j++) {
			List<Double> inputList = new ArrayList<Double>();
			List<Double> outputList = new ArrayList<Double>();
			d2 = j * (1.0D - d1) / 500.0D;

			for (int i = 0; i < inputCount; i++) {
				d3 = d2 + delta * i;
				inputList.add(InputSample.getInstance().compute(ViewerFX.getSelectedSample(), d3));
				// inputList.add((Math.sin(d3 * 3.0D * 3.141592653589793D) +
				// Math.sin(d3 * 7.0D * 3.141592653589793D) + Math.sin(d3 * 8.0D
				// * 3.141592653589793D)) / 8.0D + 0.5D);
			}
			for (int i = 0; i < outputCount; i++) {
				d3 = d2 + delay + delta * (i + inputCount - 1);
				outputList.add(InputSample.getInstance().compute(ViewerFX.getSelectedSample(), d3));
			}
			DataSeries.getInstance().addINPUTS(inputList);
			DataSeries.getInstance().addIDEALS(outputList);
		}
	}

	public static List<String> getSheetsName(String filePath) {
		List<String> sheetsName = new ArrayList<String>();
		FileInputStream fis = null;
		try {

			fis = new FileInputStream(filePath);
			HSSFWorkbook workbook = new HSSFWorkbook(fis);
			HSSFSheet sheet = null;
			for (int idx = 0; idx < workbook.getNumberOfSheets(); idx++) {
				sheet = workbook.getSheetAt(idx);
				sheetsName.add(sheet.getSheetName());
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (fis != null) {
				try {
					fis.close();
					fis = null;
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		return sheetsName;
	}

	public static void setFileSample(ITester tester, String filePath, Integer sheetIndex) throws Exception {

		DataSeries.getInstance().clearSeries();
		FileInputStream fis = null;

		try {
			
			initContext();

			fis = new FileInputStream(filePath);

			HSSFWorkbook workbook = new HSSFWorkbook(fis);
			workbook.setMissingCellPolicy(Row.CREATE_NULL_AS_BLANK);
			HSSFSheet sheet = workbook.getSheetAt(sheetIndex - 1);
			String lastContentType = null;
			Iterator rows = sheet.rowIterator();
			int rowIdx = 0;
			while (rows.hasNext()) {
				HSSFRow row = (HSSFRow) rows.next();

				Iterator cells = row.cellIterator();
				while (cells.hasNext()) {
					
					HSSFCell cell = (HSSFCell) cells.next();
					
					if(cell.getColumnIndex() > 0)
						continue;
					
					String contentType = cell.getStringCellValue().toUpperCase();
					int valuesSize = row.getLastCellNum() - 1;

					if (lastContentType != null && !lastContentType.contains("DATA") && contentType.contains("DATA")){
						createNetwork(tester, sheet, rowIdx);
					}

					if (contentType.equals("DATA")) {
						processData(row, valuesSize);
						lastContentType = contentType;
						break;
					} else if(contentType.equals("DATASIMPLEIMG")){
						processSimpleImage(row, valuesSize);
						lastContentType = contentType;
						break;
					} else if(contentType.equals("DATAIMG")){
						processImage(row, valuesSize);
						lastContentType = contentType;
						break;
					} else if(contentType.equals("DATAIMGTEST")){
						processImageTest(row, valuesSize);
						lastContentType = contentType;
						break;
					}else if (contentType.equals("TEST")){
						processTest(row, valuesSize);
						lastContentType = contentType;
						break;
					} else if (contentType.equals("NETWORKTYPE")) {
						processNetworkType(cells, valuesSize);
					} else if (contentType.equals("KIND")) {
						processKind(cells, valuesSize);
					} else if (contentType.equals("LAYER")) {
						processLayer(cells, valuesSize);
					} else if (contentType.equals("AREASCALE")) {
						processAreaScale(cells, valuesSize);
					} else if (contentType.equals("AREATYPE")) {
						processAreaType(cells, valuesSize);
					} else if (contentType.equals("AREAIMAGE")) {
						processAreaImage(cells, valuesSize);
					} else if (contentType.equals("AREA")) {
						processArea(cells, valuesSize);
					} else if (contentType.equals("NODECOUNT")) {
						processNodeCount(cells, valuesSize);
					} else if (contentType.equals("NODEBIASWEIGHT")) {
						processNodeBiasWeight(cells, valuesSize);
					} else if (contentType.equals("NODETYPE")) {
						processNodeType(cells, valuesSize);
					} else if (contentType.equals("NODEACTIVATION")) {
						processNodeActivation(cells, valuesSize);
					} else if (contentType.equals("NODERECURRENT")) {
						processNodeRecurrent(cells, valuesSize);
					} else if (contentType.equals("LINK")) {
						processLink(cells, valuesSize);
					} else if (contentType.equals("LINKAGE")) {
						processLinkage(cells, valuesSize);
					} else if (contentType.equals("LINKAGEBETWEENAREAS")) {
						processLinkageBetweenAreas(cells, valuesSize);
					} else if (contentType.equals("LINKAGETARGETEDAREA")) {
						processLinkageTargetedArea(cells, valuesSize);						
					} else if (contentType.equals("LINKAGEOPTPARAMS")) {
						processLinkageOptParams(cells, valuesSize);
					} else if (contentType.equals("LINKAGEWEIGHTMODIFIABLE")) {
						processLinkageWeightModifiable(cells, valuesSize);
					} else if (contentType.equals("FILTER")) {
						processFilter(cells, valuesSize);
					} else if (contentType.equals("LABEL")) {
						processLabel(cells, valuesSize);
					} else if (contentType.equals("AREASAMPLING")){
						processSampling(cells, valuesSize);
					}

					lastContentType = contentType;
				}

				rowIdx++;
				// sheetData.add(data);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (fis != null) {
				try {
					fis.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}

	private static void processSimpleImage(HSSFRow row, int valuesSize) {
		
		// Loading of image
		HSSFCell cellImgPath = (HSSFCell) row.getCell(1);
		Image image = new Image("file:" + cellImgPath.getRichStringCellValue().getString());
		
		PixelReader pixelReader = image.getPixelReader();
		List<Double> dataInput = null;
		List<Double> dataIdeal = null;        
        // Determine the color of each pixel in the image
        for (int readY = 0; readY < image.getHeight(); readY = readY + 4) {
            for (int readX = 0; readX < image.getWidth(); readX = readX + 4) {
                Color color = pixelReader.getColor(readX, readY);
		
				dataInput = new ArrayList<Double>(getContext().getNodeSumByLayerAndKind(0, "INPUT"));
				dataIdeal = new ArrayList<Double>(getContext().getNodeSumByKind("OUTPUT"));
				// while(cells.hasNext()){
				for (int indCol = 1; indCol <= valuesSize; indCol++) {
					HSSFCell cell = (HSSFCell) row.getCell(indCol);
					if (cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC || cell.getCellType() == HSSFCell.CELL_TYPE_BLANK 
							|| cell.getCellType() == HSSFCell.CELL_TYPE_FORMULA || cell.getCellType() == HSSFCell.CELL_TYPE_STRING) {
						
						
						if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("INPUT") != -1) {
							if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("X"))
								dataInput.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), (double) readX / 100));
							else if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("Y"))
								dataInput.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), (double) readY / 100));
							else if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("OPACITY"))
								dataInput.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), color.getOpacity()));
								
						} 
						else if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("OUTPUT") != -1) {
							if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("RED"))
								dataIdeal.add(color.getRed());
							else if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("GREEN"))
								dataIdeal.add(color.getGreen());
							else if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("BLUE"))
								dataIdeal.add(color.getBlue());
							else if(getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("OPACITY"))
								dataIdeal.add(color.getOpacity());
							else{
								double value = cell.getNumericCellValue();
								dataIdeal.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), value));
							}
						}
					}
				}
		
				DataSeries.getInstance().addINPUTS(dataInput);
				DataSeries.getInstance().addIDEALS(dataIdeal);
		
            }
        }
		
	}
	
	private static String getPath(String imgPath) {
		
		// Détection du système d'exploitation
		String osName = System.getProperty("os.name").toLowerCase();
		String formattedPath;

		if (osName.contains("win")) {
		    // Sur Windows, les chemins peuvent commencer par un lecteur, donc on les laisse tels quels
		    formattedPath = "file:./" + imgPath.replace("\\", "/");
		} else {
		    // Sur macOS et Linux, on s'assure simplement que le chemin commence par "file:"
		    formattedPath = "file:./" + imgPath.replace("\\", "/");
		}
		
		return formattedPath;
	}
	
	private static void processImage(HSSFRow row, int valuesSize) throws Exception {
		
		// Loading of image
		HSSFCell cellImgPath = (HSSFCell) row.getCell(1);
		
		String imgPath = cellImgPath.getRichStringCellValue().getString();

		// Création de l'objet Image avec le chemin formaté
		Image image = new Image(getPath(imgPath));		
		
		if(image.getException() != null)
			throw image.getException();
		
		PixelReader pixelReader = image.getPixelReader();
		List<Double> dataInput = new ArrayList<Double>(getContext().getNodeSumByLayerAndKind(0, "INPUT"));
		List<Double> dataIdeal = new ArrayList<Double>(getContext().getNodeSumByKind("OUTPUT"));
		
		for (int indCol = 1; indCol <= valuesSize; indCol++) {
			
			HSSFCell cell = (HSSFCell) row.getCell(indCol);
			
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			
			if (cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC
					|| cell.getCellType() == HSSFCell.CELL_TYPE_FORMULA || cell.getCellType() == HSSFCell.CELL_TYPE_STRING) {

				if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("INPUT") != -1) {
						
						// Determine the color of each pixel in the image
						Color color = null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataInput.add(getInstance().compute(
										getContext().getFilter(cell.getColumnIndex() - 1), 
										color.getOpacity(), 
										image.getWidth(),
										image.getHeight(),
										Double.valueOf(readX),
										Double.valueOf(readY)));
							}
						}
						

				} else if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("OUTPUT") != -1) {
					
					if(cell.getCellType() == HSSFCell.CELL_TYPE_STRING){

						// Création de l'objet Image avec le chemin formaté
						image = new Image(getPath(cell.getRichStringCellValue().getString()));	
						if(image.getException() != null)
							throw image.getException();

						pixelReader = image.getPixelReader();
						
					}
					
					
					if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("RED")) {
						// Determine the color of each pixel in the image
						Color color = null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getRed());
							}
						}
					} else if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("GREEN")) {
						// Determine the color of each pixel in the image
						Color color =  null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getGreen());
							}
						}
					} else if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("BLUE")) {
						// Determine the color of each pixel in the image
						Color color =  null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getBlue());
							}
						}
					} else if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("SATURATION")) {
						// Determine the color of each pixel in the image
						Color color =  null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getSaturation());
							}
						}
					} else if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("OPACITY")) {
						// Determine the color of each pixel in the image
						Color color =  null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getOpacity());
							}
						}
					} else if (getContext().getLabel(cell.getColumnIndex() - 1).equalsIgnoreCase("BRIGHTNESS")) {
						// Determine the color of each pixel in the image
						Color color =  null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataIdeal.add(color.getBrightness());
							}
						}
					}  else {
						double valueNumeric = cell.getNumericCellValue();
						dataIdeal.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), valueNumeric));
					}
					
				}
			}
		}
		
		DataSeries.getInstance().addINPUTS(dataInput);
		DataSeries.getInstance().addIDEALS(dataIdeal);
		
				
	}
	
	private static void processImageTest(HSSFRow row, int valuesSize) {
		
		// Loading of image
		HSSFCell cellImgPath = (HSSFCell) row.getCell(1);
		Image image = new Image(getPath(cellImgPath.getRichStringCellValue().getString()));	
		
		PixelReader pixelReader = image.getPixelReader();
		List<Double> dataInput = new ArrayList<Double>(getContext().getNodeSumByLayerAndKind(0, "INPUT"));
		
		for (int indCol = 1; indCol <= valuesSize; indCol++) {
			HSSFCell cell = (HSSFCell) row.getCell(indCol);
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			
			if (cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC 
					|| cell.getCellType() == HSSFCell.CELL_TYPE_FORMULA || cell.getCellType() == HSSFCell.CELL_TYPE_STRING) {


				if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("INPUT") != -1) {
					
						// Determine the color of each pixel in the image
						Color color = null;
						for (int readY = 0; readY < image.getHeight(); readY = readY + getContext().getSamplings(0)) {
							for (int readX = 0; readX < image.getWidth(); readX = readX + getContext().getSamplings(0)) {
								color = pixelReader.getColor(readX, readY);
								dataInput.add(getInstance().compute(null, color.getOpacity()));
							}
						}
						

				}
			}
		}
		
		DataSeries.getInstance().addTests(dataInput);
		
				
	}
	
	private static void processTest(HSSFRow row, int valuesSize) {
		
		List<Double> dataInput = new ArrayList<Double>(getContext().getNodeSumByLayerAndKind(0, "INPUT"));
		for (int indCol = 1; indCol <= valuesSize; indCol++) {
			HSSFCell cell = (HSSFCell) row.getCell(indCol);
			if (cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC || cell.getCellType() == HSSFCell.CELL_TYPE_BLANK || cell.getCellType() == HSSFCell.CELL_TYPE_FORMULA) {
				double value = cell.getNumericCellValue();
				if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("INPUT") != -1) {
					dataInput.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), value));
				} 
			}
		}

		DataSeries.getInstance().addTests(dataInput);
		
	}

	private static void processLabel(Iterator cells, int size) {
		
		getContext().initLabels(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;

			getContext().addLabel(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}
	}
	
	private static void processSampling(Iterator cells, int valuesSize) {
		
		getContext().initSampling(valuesSize);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;

			getContext().addSample((int) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}
	}
	
	private static void processAreaScale(Iterator cells, int valuesSize) {
		
		getContext().initScale(valuesSize);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;

			getContext().addScale((int) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}
	}
	

	private static void createNetwork(ITester tester, Sheet sheet, int rowDataIdx) {

		tester.setTrainingVectorNumber(sheet.getLastRowNum() - rowDataIdx + 1);

		// tester.createNetwork("NN-" + sheetName + "-" + f.format(new Date()));
		tester.createXLSNetwork("NN-" + sheet.getSheetName() + "-", getContext());
		tester.getNetwork().finalizeConnections();

	}

	private static void processLink(Iterator cells, int size) {
		Integer idxSource = null;
		Integer idxTarget = null;
		
		getContext().initLinks(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			
			if (cell.getStringCellValue().toUpperCase().equals("SOURCE")) {
				idxSource = cell.getColumnIndex();
			} else if (cell.getStringCellValue().toUpperCase().equals("TARGET")) {
				idxTarget = cell.getColumnIndex();
			} else if (cell.getStringCellValue().toUpperCase().equals("SELF")) {
				idxSource = cell.getColumnIndex();
				idxTarget = idxSource;
			}
		}
		if (idxSource != null && idxTarget != null)
			getContext().addLink(idxSource, idxTarget);

	}
	
	private static void processLinkage(Iterator cells, int size) {
		
		getContext().initLinkages(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addLinkage(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}

	}
	
	private static void processLinkageBetweenAreas(Iterator cells, int size) {
		
		getContext().initLinkageBetweenAreas(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addLinkageBetweenAreas(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}

	}
	
	private static void processLinkageTargetedArea(Iterator cells, int size) {
		
		getContext().initLinkageTargetedArea(size);
		
		String[] areaIds = null;
		Integer[] areaIdsInteger = null;
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			
			areaIds = cell.getStringCellValue().split(";");
			
			areaIdsInteger = StreamUtils.convertArray(areaIds, Integer::parseInt, Integer[]::new);
			
			getContext().addLinkageTargetedArea(areaIdsInteger, cell.getColumnIndex() - 1);
		}

	}
	
	
	private static void processLinkageOptParams(Iterator cells, int size) {
		
		getContext().initLinkageOptParams(size);
		
		String[] params = null;
		Double[] paramsDouble = null;
		
		while (cells.hasNext()) {
			
			HSSFCell cell = (HSSFCell) cells.next();
			
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			
			params = cell.getStringCellValue().split(";");
			
			paramsDouble = StreamUtils.convertArray(params, Double::parseDouble, Double[]::new);
			getContext().addLinkageOptParams(paramsDouble, cell.getColumnIndex() - 1);
		}

	}
	
	
	private static void processLinkageWeightModifiable(Iterator cells, int size) {
		
		getContext().initLinkageWeightModifiables(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addLinkageWeightModifiable(EBoolean.valueOf(cell.getStringCellValue().toUpperCase()), cell.getColumnIndex() - 1);
		}

	}

	private static void processNodeRecurrent(Iterator cells, int size) {
		
		
		getContext().initNodeRecurrents(size);


		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
				
			getContext().addNodeRecurrent("RECURRENT".equalsIgnoreCase(cell.getStringCellValue().toUpperCase()), cell.getColumnIndex() - 1);
		}

	}

	private static void processNodeActivation(Iterator cells, int size) {
		
		getContext().initNodeActivations(size);
		

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addNodeActivation(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}

	}

	private static void processNodeType(Iterator cells, int size) {
		
		getContext().initNodeTypes(size);


		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addNodeType(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}

	}

	private static void processNodeCount(Iterator cells, int size) {
		
		getContext().initNodes(size);


		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addNode((int) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}

	}

	private static void processNodeBiasWeight(Iterator cells, int size){
		
		getContext().initNodeBiasWeight(size);
		
		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addNodeBiasWeight((double) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}
	}
	
	private static void processAreaImage(Iterator cells, int size) {

		getContext().initAreasImage(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addAreaImage(EBoolean.valueOf(cell.getStringCellValue().toUpperCase()), cell.getColumnIndex() - 1);
		}

	}
	
	private static void processAreaType(Iterator cells, int size) {

		getContext().initAreasType(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addAreaType(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}

	}
	
	private static void processArea(Iterator cells, int size) {

		getContext().initAreas(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addArea((int) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}

	}

	private static void processLayer(Iterator cells, int size) {

		getContext().initLayers(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addLayer((int) cell.getNumericCellValue(), cell.getColumnIndex() - 1);
		}

	}
	
	
	private static void processNetworkType(Iterator cells, int size) {

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addNetworkType(cell.getStringCellValue().toUpperCase());
		}
	}

	private static void processKind(Iterator cells, int size) {

		getContext().initKinds(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			getContext().addKind(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}
	}

	private static void processFilter(Iterator cells, int size) {
		
		getContext().initFilters(size);

		while (cells.hasNext()) {
			HSSFCell cell = (HSSFCell) cells.next();
			if(HSSFCell.CELL_TYPE_BLANK == cell.getCellType())
				continue;
			if(!"".equals(cell.getStringCellValue().trim()))
				getContext().addFilter(cell.getStringCellValue().toUpperCase(), cell.getColumnIndex() - 1);
		}
	}

	private static void processData(HSSFRow row, int size) {

		List<Double> dataInput = new ArrayList<Double>(getContext().getNodeSumByLayerAndKind(0, "INPUT"));
		List<Double> dataIdeal = new ArrayList<Double>(getContext().getNodeSumByKind("OUTPUT"));
		// while(cells.hasNext()){
		for (int indCol = 1; indCol <= size; indCol++) {
			
			HSSFCell cell = (HSSFCell) row.getCell(indCol);
			if (cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC  || cell.getCellType() == HSSFCell.CELL_TYPE_FORMULA) {
				
				double value = cell.getNumericCellValue();
				
				if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("INPUT") != -1) {
					dataInput.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), value));
				} else if (getContext().getKind(cell.getColumnIndex() - 1).indexOf("OUTPUT") != -1) {
					dataIdeal.add(getInstance().compute(getContext().getFilter(cell.getColumnIndex() - 1), value));
				}
			}
		}

		DataSeries.getInstance().addINPUTS(dataInput);
		DataSeries.getInstance().addIDEALS(dataIdeal);
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public ESamples getSample() {
		return sample;
	}

	public void setSample(ESamples sample) {
		this.sample = sample;
	}

	public int getFileSheetIdx() {
		return fileSheetIdx;
	}

	public void setFileSheetIdx(int fileSheetIdx) {
		this.fileSheetIdx = fileSheetIdx;
	}

}
