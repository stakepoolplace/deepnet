package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;

public class ContourLinkage extends Linkage {

	
	public ContourLinkage() {
		super();
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode) {

		// somme des entrees pondérées
		Double sigmaWI = 0D;


		PixelNode pxNode = (PixelNode) thisNode;

		sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY()).getComputedOutput() * -8D;

		if (pxNode.getX() >= 0)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY()).getComputedOutput();// * 1D;

		if (pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY()).getComputedOutput();// * 1D;

		if (pxNode.getY() >= 0)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY() - 1).getComputedOutput();// * 1D;

		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY() + 1).getComputedOutput();// * 1D;

		// diagonales

		// en haut à gauche
		if (pxNode.getX() >= 0 && pxNode.getY() >= 0)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY() - 1 ).getComputedOutput();// * 1D;

		// en haut à droite
		if (pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1 && pxNode.getY() >= 0)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY() - 1 ).getComputedOutput();// * 1D;

		// en bas à gauche
		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1 && pxNode.getX() >= 0)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY() + 1 ).getComputedOutput();// * 1D;

		// en bas à droite
		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1 && pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1)
			sigmaWI += getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY() + 1 ).getComputedOutput();// * 1D;
		
		sigmaWI -= thisNode.getBiasWeightValue();


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
		 * 1  1  1 
		 * 1 -8  1 
		 * 1  1  1
		 */
		
		PixelNode pxNode = (PixelNode) thisNode;

		getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY()).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), -8D);

		if (pxNode.getX() >= 0)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY()).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		if (pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY()).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		if (pxNode.getY() >= 0)
			getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY() - 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1)
			getCheckedPixelNodeXY(pxNode, pxNode.getX(), pxNode.getY() + 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		// diagonales

		// en haut à gauche
		if (pxNode.getX() >= 0 && pxNode.getY() >= 0)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY() - 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		// en haut à droite
		if (pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1 && pxNode.getY() >= 0)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY() - 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		// en bas à gauche
		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1 && pxNode.getX() >= 0)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() - 1, pxNode.getY() + 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

		// en bas à droite
		if (pxNode.getY() <= pxNode.getAreaSquare().getHeightPx() - 1 && pxNode.getX() <= pxNode.getAreaSquare().getWidthPx() - 1)
			getCheckedPixelNodeXY(pxNode, pxNode.getX() + 1, pxNode.getY() + 1).link(pxNode, ELinkType.REGULAR, isWeightModifiable(), 1D);

	}

	private IPixelNode getCheckedPixelNodeXY(INode thisNode, int x, int y) {

		IPixelNode node = null;
		IPixelNode pxNode = (PixelNode) thisNode;
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();

		node = subArea.getNodeXY(x, y);

		if(node == null) {

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
