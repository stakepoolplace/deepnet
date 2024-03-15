package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public class LOGStaticLinkage extends LaplacianOfGaussianLinkage {

	// Coefficients issus du calcul avec :
	//	double N = 3D;
	//	double deltaX = 1;
	//	double k = 4D;
	//	double Ox = (N-1) * deltaX / k;
	//	double Oy = Ox;
	//	double alpha = 1D;
	//	double Mu = 0D;
	
	
	// Matrice carrée d'ordre N
	private static int N = 5;
	
	private static Double[][] staticFilter = new Double[][]{
																	{ 0.00, 0.00, 0.01, 0.00, 0.00 },
																	{ 0.00, 0.28, 0.69, 0.28, 0.00 },
																	{ 0.01, 0.69, -5.09, 0.69, 0.01},
																	{ 0.00, 0.28, 0.69, 0.28, 0.00 },
																	{ 0.00, 0.00, 0.01, 0.00, 0.00 }};
																	
																	

	public LOGStaticLinkage() {
	}
	
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		SigmaWi sigmaWI = new SigmaWi();
			
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_LOG_STATIC, null, (IPixelNode) thisNode, subArea);
		
		subArea.applyConvolutionFilter(this, ID_FILTER_LOG_STATIC, (IPixelNode) thisNode, sigmaWI);
			
		
		return sigmaWI.value();
	}
	
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_LOG_STATIC, null, (IPixelNode) thisNode, subArea);
		
		subArea.applyConvolutionFilter(this, ID_FILTER_LOG_STATIC, (IPixelNode) thisNode, 0.0001f);
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		int x = ((IPixelNode) sublayerNode).getX();
		int y = ((IPixelNode) sublayerNode).getY();
		
		int centerX = subArea.getNodeCenterXY().getX();
		int centerY = subArea.getNodeCenterXY().getY();
		
		int Xmat = x - centerX + ( (N-1) / 2 );
		int Ymat = y - centerY + ( (N-1) / 2 );
		
		if(Xmat >= 0 && Xmat <= (N-1) && Ymat >= 0 && Ymat <= (N-1)){
			return staticFilter[Xmat][Ymat];
		}
		
		return 0D;
	}

}
