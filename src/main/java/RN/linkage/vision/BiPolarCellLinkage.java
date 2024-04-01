package RN.linkage.vision;

import RN.IAreaSquare;
import RN.ILayer;
import RN.linkage.Linkage;
import RN.links.ELinkType;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;

/**
 * @author Eric Marchand
 *
 */
public class BiPolarCellLinkage extends Linkage{

	
	public BiPolarCellLinkage() {
		super();
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode) {

		Double sigmaWI = 0D;
		
		Double sign = -1D;

		// somme des entrees pondérées
		ILayer sublayer = thisNode.getArea().getPreviousLayer();
		if (sublayer != null) {
			
			PixelNode bipolarNode = (PixelNode) thisNode;
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				sign = 1D;

				sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY()).getComputedOutput() * 8D * -sign;
			
			
			if (bipolarNode.getX() >= 0){
				
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY()).getComputedOutput() * sign;
			}

			if (bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY()).getComputedOutput() * sign;
			}
			
			if (bipolarNode.getY() >= 0){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() - 1).getComputedOutput() * sign;
			}
			
			if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() + 1).getComputedOutput() * sign;
			}
			
			// diagonales

			// en haut à gauche
			if (bipolarNode.getX() >= 0 && bipolarNode.getY() >= 0){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() - 1 ).getComputedOutput() * sign;
			}
			
			// en haut à droite
			if (bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1 && bipolarNode.getY() >= 0){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() - 1 ).getComputedOutput() * sign;
			}
			
			// en bas à gauche
			if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1 && bipolarNode.getX() >= 0){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() + 1 ).getComputedOutput() * sign;
			}
			
			// en bas à droite
			if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1 && bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1){
					sigmaWI += getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() + 1 ).getComputedOutput() * sign;
			}
			
			sigmaWI -= thisNode.getBiasWeightValue();
			
		}

		return sigmaWI;
	}
	
	/* (non-Javadoc)
	 * @see RN.linkage.FullFanOutLinkage#sublayerFanOutLinkage(RN.Layer)
	 */
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {

		/**
		 * Filtre délimiteur de contours Implémentation de l'opérateur Laplacien
		 * avec diagonales
		 * 
		 * Voie ON
		 * -1  -1  -1 
		 * -1   8  -1 
		 * -1  -1  -1
		 * 
		 * Voie OFF
		 *  1   1   1 
		 *  1  -8   1 
		 *  1   1   1
		 */
		
		PixelNode bipolarNode = (PixelNode) thisNode;

		if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
			getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY()).link(bipolarNode, ELinkType.ON, isWeightModifiable(), 8D);
		
		if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
			getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY()).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), -8D);

		if (bipolarNode.getX() >= 0){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY()).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY()).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		if (bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY()).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY()).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		if (bipolarNode.getY() >= 0){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() - 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() - 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() + 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX(), bipolarNode.getY() + 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		// diagonales

		// en haut à gauche
		if (bipolarNode.getX() >= 0 && bipolarNode.getY() >= 0){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() - 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() - 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		// en haut à droite
		if (bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1 && bipolarNode.getY() >= 0){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() - 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() - 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		// en bas à gauche
		if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1 && bipolarNode.getX() >= 0){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() + 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() - 1, bipolarNode.getY() + 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

		// en bas à droite
		if (bipolarNode.getY() <= bipolarNode.getAreaSquare().getHeightPx() - 1 && bipolarNode.getX() <= bipolarNode.getAreaSquare().getWidthPx() - 1){
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_L)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() + 1).link(bipolarNode, ELinkType.ON, isWeightModifiable(), -1D);
			
			if(bipolarNode.getNodeType() == ENodeType.BIPOLAR_S)
				getCheckedPixelNodeXY(bipolarNode, sublayer, bipolarNode.getX() + 1, bipolarNode.getY() + 1).link(bipolarNode, ELinkType.OFF, isWeightModifiable(), 1D);
		}

	}

	private IPixelNode getCheckedPixelNodeXY(INode thisNode, ILayer sublayer, int x, int y) {

		IPixelNode node = null;
		IPixelNode pxNode = (IPixelNode) thisNode;
		
		IAreaSquare subArea = (IAreaSquare) sublayer.getAreas().get(0);

		try {
			node = subArea.getNodeXY(x, y);
		} catch (Exception e) {

			// on se trouve à proximité d'un bord, on recopie le pixel du bord
			try {
				
				if(x < 0)
					x += 1;
				if(y < 0)
					y += 1;
				if(x > pxNode.getAreaSquare().getWidthPx() - 1)
					x -= 1;
				if(y > pxNode.getAreaSquare().getHeightPx() - 1)
					y -= 1;
					
				node = subArea.getNodeXY(x, y);
				
			} catch (Exception e1) {
				e1.printStackTrace();
			}

		}

		return node;

	}

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParameters() {
		// TODO Auto-generated method stub
		
	}
	
}
