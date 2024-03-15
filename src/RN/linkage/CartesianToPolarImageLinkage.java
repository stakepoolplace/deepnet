package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.Coordinate;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import javafx.scene.layout.Pane;

public class CartesianToPolarImageLinkage extends FilterLinkage {
	

	//Number of orientations (between 3 to 20) 8 is a typical value
	private Double n_t = null;
	
	//Number of scales of the multiresolution scheme
	private Double n_s = null;
	
	private Double angle = 360D;
	
	private Double base = null;
	
	private Integer radius = null;
	


	public CartesianToPolarImageLinkage() {
	}
	

	
	public void initParameters() {
		
		if(params.length != 2 && params.length != 3)
			throw new RuntimeException("Missing Cartesian to Polar parameters'");
		
		
		n_t = params[0];
		n_s = params[1];
		
		if(params.length == 3 && params[2] != null)
			angle = params[2];
		
		base = 1 + (Math.PI / (Math.sqrt(3D) * n_t));
		

	}

	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		radius = Math.min(Math.round(subArea.getHeightPx() / 2), Math.round(subArea.getWidthPx() / 2)) - 1;
		
		
		
		double nbPixPerDegree;
		double nbPixPerRadius;
		double a;
		double r;
		
		Coordinate coord = new Coordinate(pix.getX(), pix.getY());
		//coord.setBase(this.base);
		coord.setX0((double) subArea.getNodeCenterX());
		coord.setY0((double) subArea.getNodeCenterY());
		
		nbPixPerDegree = pix.getAreaSquare().getWidthPx() / angle;
		a = coord.getX() / nbPixPerDegree;
		
		nbPixPerRadius = pix.getAreaSquare().getHeightPx() / radius;
		r = coord.getY() / nbPixPerRadius;
		
		coord.setTheta(Math.toRadians(a));
		coord.setR(r);
		coord.polarToLinearSystem();
		
		IPixelNode subPix = subArea.getNodeXY(coord.getX().intValue(), coord.getY().intValue());
		if(subPix != null){
			sigmaWI.sum(subPix.getComputedOutput());
		}
		
		
		return sigmaWI.value();
	}
	
	private void cartesianToPolar(IPixelNode pix, IAreaSquare subArea) {
		
		int centerX = Math.round(pix.getAreaSquare().getHeightPx() / 2);
		int centerY = Math.round(pix.getAreaSquare().getWidthPx() / 2);
		int i = 1;
		
		int x, y;
		
		for(int r=0; r < this.radius; r++){
			int j = 1;
			for(int a=0; a < 2 * Math.PI - 2 * Math.PI / angle; a +=2 * Math.PI / angle ){
				x = (int) (centerX + Math.round(r * Math.cos(a)));
				y = (int) (centerY + Math.round(r * Math.sin(a)));
				
				pix.getAreaSquare().getNodeXY(i, j).setComputedOutput(subArea.getNodeXY(x, y).getComputedOutput());
				j++;
			}
			i++;
		}
		

	}
	
//	   img         = double(img);
//	   [rows,cols] = size(img);
//	   cy          = round(rows/2);
//	   cx          = round(cols/2);
//	   
//	   if exist('radius','var') == 0
//	      radius = min(round(rows/2),round(cols/2))-1;
//	   end
//	   
//	   if exist('angle','var') == 0
//	      angle = 360;
//	   end
//	  
//	   pcimg = [];
//	   i     = 1;
//	   
//	   for r=0:radius
//	      j = 1;
//	      for a=0:2*pi/angle:2*pi-2*pi/angle
//	         pcimg(i,j) = img(cy+round(r*sin(a)),cx+round(r*cos(a)));
//	         j = j + 1;
//	      end
//	      i = i + 1;
//	   end
//	end



	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		radius = Math.min(Math.round(subArea.getHeightPx() / 2), Math.round(subArea.getWidthPx() / 2)) - 1;
		
		double nbPixPerDegree;
		double nbPixPerRadius;
		double a;
		double r;
		
		Coordinate coord = new Coordinate(pix.getX(), pix.getY());
		//coord.setBase(this.base);
		coord.setX0((double) subArea.getNodeCenterX());
		coord.setY0((double) subArea.getNodeCenterY());
		
		nbPixPerDegree = pix.getAreaSquare().getWidthPx() / angle;
		a = coord.getX() / nbPixPerDegree;
		
		nbPixPerRadius = pix.getAreaSquare().getHeightPx() / radius;
		r = coord.getY() / nbPixPerRadius;
		
		coord.setTheta(Math.toRadians(a));
		coord.setR(r);
		coord.polarToLinearSystem();
		
		IPixelNode subPix = subArea.getNodeXY(coord.getX().intValue(), coord.getY().intValue());
		if(subPix != null){
			subPix.link(thisNode, ELinkType.REGULAR, isWeightModifiable());
		}
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre Log-Gabor
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				getN_t(),
				getN_s(),
				(double) (sublayerNode.getX() - subArea.getWidthPx() / 2D) * 4D,
				(double) (sublayerNode.getY() - subArea.getHeightPx() / 2D) * 4D
				);
		
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
	
	public void addGraphicInterface(Pane pane) {
		
	}


	public Double getN_t() {
		return n_t;
	}



	public void setN_t(Double n_t) {
		this.n_t = n_t;
	}



	public Double getN_s() {
		return n_s;
	}



	public void setN_s(Double n_s) {
		this.n_s = n_s;
	}



}
