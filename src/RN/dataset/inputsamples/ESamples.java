package RN.dataset.inputsamples;

/**
 * @author Eric Marchand
 *
 */
public enum ESamples {
	ADD, SUBSTRACT,
	SINUS, COSINUS,
	// Champs recepteurs vision, gaussienne, gaussienne De Marr (principe d'Heisenberg), dérivée première, dérivée seconde (filtre LOG) ...
	GAUSSIAN, GAUSSIAN_DE_MARR, Gx_DE_MARR, Gy_DE_MARR, 
	G_D1xy_DE_MARR, G_Dxy_DE_MARR, G_Dx_DE_MARR, G_Dy_DE_MARR, 
	G_D2xy_DE_MARR, G_D2xyTheta_DE_MARR, G_Dxx_DE_MARR, G_Dyy_DE_MARR,  
	G_D3xy_DE_MARR, G_Dxxx_DE_MARR, G_Dxxy_DE_MARR, G_Dxyy_DE_MARR, G_Dyyy_DE_MARR,
	G_COURBURE_DE_MARR,
	COMPLEX, 
	CHAOS, 
	MEANEXP1, MEANEXP2, MEANEXP3, 
	IDENTITY, 
	FILE, MACD, TIMESERIE, RAND, NONE, SAMPLING, MULTIPLY,
	LOG_GABOR;
	
}
