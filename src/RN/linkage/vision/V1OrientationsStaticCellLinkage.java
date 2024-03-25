package RN.linkage.vision;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public class V1OrientationsStaticCellLinkage extends V1OrientationsCellLinkage {


	public V1OrientationsStaticCellLinkage() {
	}
	
	
	// Matrice carrée d'ordre N
	private static int N = 3;
	
	private static Double[][] staticFilter = new Double[][]{
																{0D,  0D,  1D,  1D,  1D,  0D,  0D},
																{0D,  1D,  1D,  1D,  1D,  1D,  0D},
																{1D,  1D, -1D, -4D, -1D,  1D,  1D},
																{1D,  1D, -4D, -8D, -4D,  1D,  1D},
																{1D,  1D, -1D, -4D, -1D,  1D,  1D},
																{0D,  1D,  1D,  1D,  1D,  1D,  0D},
																{0D,  0D,  1D,  1D,  1D,  0D,  0D},
	};
	
	public void initParameters() {
		
		setTheta(params[0]);
		
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		if(getTheta() == null)
			throw new RuntimeException("Theta is not set on area");
		
		initFilter(this, ID_FILTER_V1Orientation, ESamples.G_D2xyTheta_DE_MARR, (IPixelNode) thisNode, subArea, getTheta());
		
		subArea.applyConvolutionFilter(this, ID_FILTER_V1Orientation, (IPixelNode) thisNode, 0.000001f);
		
	}
}
