package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public class DOGStaticLinkage extends DifferenceOfGaussianLinkage {

	
	// Matrice carrée d'ordre N
	private static int N = 7;
	
	private static Double[][] staticFilter = new Double[][]{
																{0D,  0D,  1D,  1D,  1D,  0D,  0D},
																{0D,  1D,  1D,  1D,  1D,  1D,  0D},
																{1D,  1D, -1D, -4D, -1D,  1D,  1D},
																{1D,  1D, -4D, -8D, -4D,  1D,  1D},
																{1D,  1D, -1D, -4D, -1D,  1D,  1D},
																{0D,  1D,  1D,  1D,  1D,  1D,  0D},
																{0D,  0D,  1D,  1D,  1D,  0D,  0D},
	};
	

	public DOGStaticLinkage() {
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		initFilter(this, ID_FILTER_DOG_STATIC, null, (IPixelNode) thisNode, subArea);
			
		subArea.applyConvolutionFilter(this, ID_FILTER_DOG_STATIC, (IPixelNode) thisNode, sigmaWI);
			
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		initFilter(this, ID_FILTER_DOG_STATIC, null, (IPixelNode) thisNode, subArea);
		
		subArea.applyConvolutionFilter(this, ID_FILTER_DOG_STATIC, (IPixelNode) thisNode, 0.001f);
	}
	
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double mu, Double alpha, Double ox, Double oy) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		int x = sublayerNode.getX();
		int y = sublayerNode.getY();
		
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
