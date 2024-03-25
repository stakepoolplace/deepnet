package RN;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.EFilterPosition;
import RN.linkage.Filter;
import RN.linkage.FilterLinkage.FilterIndex;
import RN.linkage.IFilterLinkage;
import RN.linkage.SigmaWi;
import RN.linkage.vision.Gradient;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.ImageNode;
import RN.nodes.Node;
import RN.nodes.PixelNode;
import javafx.scene.paint.Color;

/**
 * @author Eric Marchand
 * 
 */public class AreaSquare extends Area implements IAreaSquare, IArea{
	
	private Integer nodeCount = null;
	
	
	protected AreaSquare(){
	}
	
	public AreaSquare(int nodeCount) {
		
		this.nodes = new ArrayList<INode>(nodeCount);
		initWidthPx(nodeCount);
		this.nodeCount = nodeCount;
	}
	

	public AreaSquare(int nodeCount, boolean showImage) {
		
		this.nodes = new ArrayList<INode>(nodeCount);
		this.showImage = showImage;
		initWidthPx(nodeCount);
		if(showImage){
			this.imageArea = new ImageNode(EActivation.IDENTITY, this.widthPx, this.heightPx);
			this.imageArea.setArea(this);
		}
		this.nodeCount = nodeCount;
	}
	
	public AreaSquare(int nodeCountX, int nodeCountY, boolean showImage) {
		int count = nodeCountX * nodeCountY;
		this.nodes = new ArrayList<INode>(count);
		this.showImage = showImage;
		initWidthHeightPx(nodeCountX, nodeCountY);
		if(showImage){
			this.imageArea = new ImageNode(EActivation.IDENTITY, this.widthPx, this.heightPx);
			this.imageArea.setArea(this);
		}
		this.nodeCount = count;
	}

	public AreaSquare(int nodeCount, boolean showImage, String comment) {
		
		this.nodes = new ArrayList<INode>(nodeCount);
		this.showImage = showImage;
		this.comment += comment;
		initWidthPx(nodeCount);
		this.nodeCount = nodeCount;
	}

	
	public PixelNode getNode(int index) {
		return (PixelNode) nodes.get(index);
	}

	@Override
	public int getNodeCount() {
		return this.nodes.size();
	}
	
	
	public Filter getFilter(int filterId){
		FilterIndex idx = new FilterIndex(getIdentification(), filterId);
		return getFilterLinkage().getFilter(idx);
	}


	@Override
	public IPixelNode getNodeXY(int x, int y) {
		try{
			int id = nodeXYToNodeId(x, y);
			if(id < 0 || id > nodes.size() - 1)
				return null;
			
			return (IPixelNode) nodes.get(id);
			
		} catch (Exception ignore) {
		}
		
		return null;
	}
	
	@Override
	public IPixelNode getNodeXY(int x, int y, int sampling) {
		
		if(sampling == 1)
			return getNodeXY(x, y);
		
		try{
			return (IPixelNode) nodes.get(nodeXYToNodeId(x, y, sampling));
		} catch (Exception ignore) {
			ignore.printStackTrace();
		}
		
		return null;
	}
	

	
	/**
	 * Retourne le node à la position (x,y) avec comme point d'origine (x0,y0) après une rotation d'angle theta
	 * @param x
	 * @param y
	 * @param theta en radian
	 */
	public IPixelNode getNodeXY(int x, int y, int x0, int y0, double theta) {
		
		IPixelNode node = null;
		
		try{
			node = (IPixelNode) nodes.get(nodeXYToNodeId(x, y, x0, y0, theta));
		} catch (Exception ignore) {
		}
		
		return node;
	}

	
	/**
	 * Return nodes in square zone 
	 * @param x0 x at top left point 
	 * @param y0 y at top left point
	 * @param width
	 * @param height
	 * @return
	 * @throws Exception
	 */
	public List<IPixelNode> getNodesInSquareZone(int x0, int y0, int width, int height) {
		
		List<IPixelNode> nodes = new ArrayList<IPixelNode>();
		IPixelNode pix = null;
		
		for(int y = y0; y < y0 + height; y++){
			for(int x = x0; x < x0 + width; x++){
				try {
					pix = getNodeXY(x,y);
					if(pix != null)
						nodes.add(pix);
				} catch (Exception ignore) {
				}
			}
		}
		return nodes;
	}
	
	/**
	 * Return nodes in a circlar zone
	 * @param x0 x center point
	 * @param y0 y center point
	 * @param radius
	 * @return
	 * @throws Exception
	 */
	public List<IPixelNode> getNodesInCirclarZone(int x0, int y0, int radius) {
		
		List<IPixelNode> nodes = new ArrayList<IPixelNode>();
		IPixelNode node = null;
		
		for(int y = y0 - radius; y <= y0 + radius; y++){
			for(int x = x0 - radius; x <= x0 + radius; x++){
				
				double distance = Math.sqrt(Math.pow(x0 - x, 2D) + Math.pow(y0 - y, 2D));
				
				if(distance <= radius){
					node = getNodeXY(x,y);
					if(node != null){
						nodes.add(node);
					}
				}
			}
		}
		
		return nodes;
	}
	
	public List<IPixelNode> getNodesOnCirclarPerimeter(int x0, int y0, int radius) {
		
		List<IPixelNode> nodes = new ArrayList<IPixelNode>();
		IPixelNode node = null;
		
		for(int y = y0 - radius; y <= y0 + radius; y++){
			for(int x = x0 - radius; x <= x0 + radius; x++){
				
				double distance = Math.sqrt(Math.pow(x0 - x, 2D) + Math.pow(y0 - y, 2D));
				
				if(Math.abs(distance - radius) < 0.5D ){
					node = getNodeXY(x,y);
					if(node != null){
						nodes.add(node);
					}
				}
			}
		}
		
		return nodes;
	}
	
	public void initWidthPx(int pixSize){
		
		Double width = Math.sqrt(pixSize);
		
		if(width == 0D)
			throw new RuntimeException("Le nombre de neurones dans la liste est vide.");
		
		if(width.intValue() - width.doubleValue() != 0)
			throw new RuntimeException("Le nombre de neurones doit être un carré d'un entier. nbr=" + pixSize);
		
		// Carré
		this.widthPx = width.intValue();
		this.heightPx = this.widthPx;
		this.nodeCenterX = widthPx / 2;
		this.nodeCenterY = heightPx / 2;
	}
	
	public void initWidthHeightPx(int pixSizeX, int pixSizeY){
		
		
		if(pixSizeX == 0)
			throw new RuntimeException("Le nombre de neurones dans la liste est vide.");
		
		
		// Rectangle
		this.widthPx = pixSizeX;
		this.heightPx = pixSizeY;
		this.nodeCenterX = widthPx / 2;
		this.nodeCenterY = heightPx / 2;
	}
	
	public Integer getWidthPx(){
		return this.widthPx;
	}
	
	public IPixelNode getNodeCenterXY() {
		return (IPixelNode) nodes.get(nodeXYToNodeId(nodeCenterX, nodeCenterY));
	}
	
	
	public Integer nodeXYToNodeId(int x, int y) {
		
		if(x < 0 || x > widthPx - 1 || y < 0 || y > heightPx - 1)
			return -1;
		
		return x + y * widthPx ;
		
	}
	
	public Integer nodeXYToNodeId(int x, int y, int sampling) {
		
		x = x * sampling;
		y = y * sampling;
		
		if(x < 0 || x > widthPx - 1 || y < 0 || y > heightPx - 1)
			return -1;
		
		return x + y * widthPx ;
		
	}
	
	public Integer nodeXYToNodeId(int x, int y, int x0, int y0, double theta) {
		
		if(x < 0 || x > widthPx - 1 || y < 0 || y > heightPx - 1
				|| x0 < 0 || x0 > widthPx - 1 || y0 < 0 || y0 > heightPx - 1)
			return -1;
		
		int newX = (int) Math.round((x-x0) * Math.cos(theta) - (y-y0) * Math.sin(theta)) + x0 ;
		int newY = (int) Math.round((x-x0) * Math.sin(theta) + (y-y0) * Math.cos(theta)) + y0 ;
		
		
		return newX + newY * widthPx ;
		
	}
	
	
	public int[] nodeIdToNodeXY(int id) {
		
		int x;
		int y;
		
		Double val = new Double(id / widthPx);
		y = val.intValue();
		x = id - y * widthPx;
		
		return new int[] { x, y };

	}
	
	
	public void applyConvolutionFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, Float ecartType){
		
		double weight = 0D;
		
		FilterIndex index = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		Filter filter = linkage.getFilter(index);
		
		IPixelNode sourceNode = null;
		
		int centerX = thisNode.getX();
		int centerY = thisNode.getY();
		int halfHeightFilter = (filter.getHeight() - 1) / 2; 
		int halfWidthFilter = (filter.getWidth() - 1) / 2;
		int y0 = Math.max(0, (centerY - halfHeightFilter) * linkage.getSampling());
		int x0 = Math.max(0, (centerX - halfWidthFilter) * linkage.getSampling());
		int ystop = Math.min(this.getHeightPx() - 1, (centerY + halfHeightFilter) * linkage.getSampling());
		int xstop = Math.min(this.getWidthPx() - 1, (centerX + halfWidthFilter) * linkage.getSampling());
		
		for(int y = y0 ; y <= ystop; y += linkage.getSampling()){
			for(int x = x0; x <= xstop; x += linkage.getSampling()){
				
				sourceNode = this.getNodeXY(x, y);
				
				weight = linkage.getFilterValue(index, EFilterPosition.CENTER, thisNode, sourceNode);
				
				if(Math.abs(weight) > ecartType){
					sourceNode.link((INode) thisNode, ELinkType.REGULAR, linkage.isWeightModifiable(), weight);
				}
				
			}
			
		}
		
	}
	
	public void applyMaxPoolingFilter(IFilterLinkage linkage, int width, int stride, IPixelNode thisNode){
		
		double value = 0D;
		
		int sampling = linkage.getSampling();
		int topLeftX = thisNode.getX();
		int topLeftY = thisNode.getY();
		int y0 = topLeftY * sampling * stride;
		int x0 = topLeftX * sampling * stride;
		int ystop = y0 + width - 1;
		int xstop = x0 + width - 1;
		
		IPixelNode sourceNode = null;
		IPixelNode maxSourceNode = null;
		
		Double maxValue = null;
		
		for(int y = y0 ; y <= ystop; y += sampling){
			for(int x = x0; x <= xstop; x += sampling){
				
				if(x >= 0 && x <= this.getWidthPx() - 1 && y >=0 && y <= this.getHeightPx() - 1){
					
					sourceNode = this.getNodeXY(x, y);
					value = sourceNode.getComputedOutput();
					
					if(maxValue == null || maxValue < value){
						maxValue = value;
						maxSourceNode = sourceNode;
					}
				}
				
			}
			
		}
		
		if(maxSourceNode != null){
			thisNode.getInputs().clear();
			maxSourceNode.link((INode) thisNode, ELinkType.REGULAR, linkage.isWeightModifiable(), maxValue);
		}
		
		
	}
	
	public void applyMaxPoolingFilter(IFilterLinkage linkage, int width, int stride, IPixelNode thisNode, SigmaWi sigmaWI){
		
		
		int sampling = linkage.getSampling();
		int topLeftX = thisNode.getX();
		int topLeftY = thisNode.getY();
		int y0 = topLeftY * sampling * stride;
		int x0 = topLeftX * sampling * stride;
		int ystop = y0 + width - 1;
		int xstop = x0 + width - 1;
		
		IPixelNode sourceNode = null;
		
		Double value = null;
		
		for(int y = y0 ; y <= ystop; y += sampling){
			for(int x = x0; x <= xstop; x += sampling){
				
				if(x >= 0 && x <= this.getWidthPx() - 1 && y >=0 && y <= this.getHeightPx() - 1){
					
					sourceNode = this.getNodeXY(x, y);
					value = sourceNode.getComputedOutput();
					
					if(value > sigmaWI.value() ){
						sigmaWI.setSigmaWi(value);
					}
				}
				
			}
			
		}
		
	}
	
	public void applyConvolutionCompositeFilter(IFilterLinkage linkage, int idFilter1, int idFilter2, ESamples op, IPixelNode thisNode, Float ecartType){
		
		double weight = 0D;
		double weight1 = 0D;
		double weight2 = 0D;
		
		FilterIndex index1 = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter1);
		FilterIndex index2 = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter2);
		Filter filter1 = linkage.getFilter(index1);
		Filter filter2 = linkage.getFilter(index2);
		
		IPixelNode sourceNode = null;
		
		int centerX = thisNode.getX();
		int centerY = thisNode.getY();
		int heightMax = Math.max(filter1.getHeight(), filter2.getHeight());
		int widthMax = Math.max(filter1.getWidth(), filter2.getWidth());
		int halfHeightFilter = (heightMax - 1) / 2; 
		int halfWidthFilter = (widthMax - 1) / 2;
		int y0 = Math.max(0, (centerY - halfHeightFilter) * linkage.getSampling());
		int x0 = Math.max(0, (centerX - halfWidthFilter) * linkage.getSampling());
		int ystop = Math.min(this.getHeightPx() - 1, (centerY + halfHeightFilter) * linkage.getSampling());
		int xstop = Math.min(this.getWidthPx() - 1, (centerX + halfWidthFilter) * linkage.getSampling());
		
		for(int y = y0 ; y <= ystop; y += linkage.getSampling()){
			for(int x = x0; x <= xstop; x += linkage.getSampling()){
				
				sourceNode = this.getNodeXY(x, y);
				
				weight1 = linkage.getFilterValue(index1, EFilterPosition.CENTER, thisNode, sourceNode);
				weight2 = linkage.getFilterValue(index2, EFilterPosition.CENTER, thisNode, sourceNode);
				
				if(op == ESamples.SUBSTRACT)
					weight = weight1 - weight2;
				else if(op == ESamples.ADD)
					weight = weight1 + weight2;
				else if(op == ESamples.MULTIPLY)
					weight = weight1 * weight2;
				
				if(Math.abs(weight) > ecartType){
					sourceNode.link((INode) thisNode, ELinkType.REGULAR, linkage.isWeightModifiable(), weight);
				}
				
			}
			
		}
		
	}
	
	public void applyConvolutionFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, SigmaWi sigmaWI) {
		
		
		FilterIndex index = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		Filter filter = linkage.getFilter(index);
		
		IPixelNode sourceNode = null;
		
		int centerX = thisNode.getX();
		int centerY = thisNode.getY();
		int halfHeightFilter = (filter.getHeight() - 1) / 2; 
		int halfWidthFilter = (filter.getWidth() - 1) / 2;
		int y0 = Math.max(0, (centerY - halfHeightFilter) * linkage.getSampling());
		int x0 = Math.max(0, (centerX - halfWidthFilter) * linkage.getSampling());
		int ystop = Math.min(this.getHeightPx() - 1, (centerY + halfHeightFilter) * linkage.getSampling());
		int xstop = Math.min(this.getWidthPx() - 1, (centerX + halfWidthFilter) * linkage.getSampling());
		
		for(int y = y0 ; y <= ystop; y += linkage.getSampling()){
			for(int x = x0; x <= xstop; x += linkage.getSampling()){
				
				sourceNode = this.getNodeXY(x, y);
				sigmaWI.sum(sourceNode.getComputedOutput() * linkage.getFilterValue(index, EFilterPosition.CENTER, thisNode, sourceNode));
			}
		}
		
		sigmaWI.sum(-thisNode.getBiasWeightValue());
		
	}
	
	

	
	public void applyConvolutionCompositeFilter(IFilterLinkage linkage, int idFilter1, int idFilter2, ESamples op, IPixelNode thisNode, SigmaWi sigmaWI){
		
		double weight = 0D;
		double weight1 = 0D;
		double weight2 = 0D;
		
		FilterIndex index1 = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter1);
		FilterIndex index2 = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter2);
		
		Filter filter1 = linkage.getFilter(index1);
		Filter filter2 = linkage.getFilter(index2);
		
		IPixelNode sourceNode = null;
		
		int sampling = linkage.getSampling();
		int centerX = thisNode.getX();
		int centerY = thisNode.getY();
		int heightMax = Math.max(filter1.getHeight(), filter2.getHeight());
		int widthMax = Math.max(filter1.getWidth(), filter2.getWidth());
		int halfHeightFilter = (heightMax - 1) / 2; 
		int halfWidthFilter = (widthMax - 1) / 2;
		int y0 = Math.max(0, (centerY - halfHeightFilter) * sampling);
		int x0 = Math.max(0, (centerX - halfWidthFilter) * sampling);
		int ystop = Math.min(this.getHeightPx() - 1, (centerY + halfHeightFilter) * sampling);
		int xstop = Math.min(this.getWidthPx() - 1, (centerX + halfWidthFilter) * sampling);
		
		for(int y = y0 ; y <= ystop; y += sampling){
			for(int x = x0; x <= xstop; x += sampling){
				
				sourceNode = this.getNodeXY(x, y);
				
				weight1 = linkage.getFilterValue(index1, EFilterPosition.CENTER, thisNode, sourceNode);
				weight2 = linkage.getFilterValue(index2, EFilterPosition.CENTER, thisNode, sourceNode);
				
				if(op == ESamples.SUBSTRACT)
					weight = weight1 - weight2;
				else if(op == ESamples.ADD)
					weight = weight1 + weight2;
				else if(op == ESamples.MULTIPLY)
					weight = weight1 * weight2;
				
				sigmaWI.sum(sourceNode.getComputedOutput() * weight);
				
			}
			
		}
		
		sigmaWI.sum(-thisNode.getBiasWeightValue());
		
	}
	
	
	
	public String toString() {
		return " Area id : " + areaId  + "    heightPx :" + heightPx + "    widthPx : " + widthPx;
	}
	
	public void showGradients(double magnitudeFactor, double magnitudeThreshold, int sampling, Color color) {
		IPixelNode pix = null;
		for(int idx = 0; idx < nodeCount; idx++){
			pix = (PixelNode) getNode(idx);
			if(pix.getX() % sampling == 0 && pix.getY() % sampling == 0){
				try {
					Gradient subGradient = null;
					double magSum = 0D;
					double avgMagnitude = 0;
					double avgTheta = 0;
					double x = 0;
					double y = 0;
					int count = 0;
					for(IPixelNode subPix : getNodesInSquareZone(pix.getX() - (sampling / 2), pix.getY() - (sampling / 2), sampling, sampling)){
						subGradient = subPix.getGradient();
						if(subGradient != null){
							magSum += subGradient.getMagnitude();
							x += Math.cos(subGradient.getTheta());
							y += Math.sin(subGradient.getTheta());
							count++;
						}
					}
					avgMagnitude = magSum / count;
					avgTheta = Math.atan2(y / count, x / count);
					avgTheta = avgTheta < 0 ? avgTheta + (2D * Math.PI) : avgTheta;
					
					if(Math.abs(avgMagnitude) > magnitudeThreshold){
						new Gradient(pix, avgTheta, avgMagnitude, null).produceGradient(magnitudeFactor, color);
					}
				} catch (Exception e) {
				}
					//gradient.produceGradient(magnitudeFactor, color);
			}
			
		}
		
	}
	

	public void imageToString(){
		
		System.out.println("Image area #"+ areaId +" : "+ getWidthPx() + " x " + getHeightPx());
		
		for (int idy = 0; idy < getHeightPx(); idy++) {
			for (int idx = 0; idx < getWidthPx(); idx++) {
				Double value = null;
				try {
					value = getNodeXY(idx, idy).getComputedOutput();
					if(value == 0D)
						System.out.print(" . ");
					else if(value < 0D)
						System.out.printf(" %.2f ", value);
					else
						System.out.printf(" %.2f ", value);
				} catch (Exception e) {
					System.out.printf(" X ");
				}

				System.out.print("\t");
			}
			System.out.print("\n");
		}
			
	}
	
	public void pixelsToString(List<IPixelNode> pixels){
		
		System.out.println("Image area #"+ areaId +" : "+ getWidthPx() + " x " + getHeightPx());
		Optional<IPixelNode> pix = null;
		Supplier<Stream<IPixelNode>> streamSupplier = () -> pixels.stream();
		for (int idy = 0; idy < getHeightPx(); idy++) {
			for (int idx = 0; idx < getWidthPx(); idx++) {
				Double value = null;
				try {
					final int x = idx;
					final int y = idy;
					pix = streamSupplier.get().filter(node -> node.getX() == x && node.getY() == y).findFirst();
					if(pix.isPresent()){
						
						value = pix.get().getComputedOutput();
						
						if(value == 0D)
							System.out.print(" . ");
						else if(value > 0D)
							System.out.printf(" %.2f ", value);
						else 
							System.out.printf(" %.2f ", value);
					
					}else
						System.out.printf(" _ ");
					
				} catch (Exception e) {
					System.out.printf(" X ");
				}

				System.out.print("\t");
			}
			System.out.print("\n");
		}
			
	}
	
	public List<Double> compareArea(AreaSquare area2){
		
		List<Double> substract = new ArrayList<Double>(nodes.size());
		
		System.out.println("Compare area #" + areaId + " - " + area2.getAreaId() + ": "+ getWidthPx() + " x " + getHeightPx());
		
		boolean equal = true;
		
		for (int idy = 0; idy < getHeightPx(); idy++) {
			for (int idx = 0; idx < getWidthPx(); idx++) {
				Double value = null;
				try {
					value = getNodeXY(idx, idy).getComputedOutput() - area2.getNodeXY(idx, idy).getComputedOutput();
					substract.add(value);
					if(value == 0D)
						System.out.print(" . ");
					else if(value < 0D){
						equal = false;
						System.out.printf(" %.2f ", value);
					}else{
						equal = false;
						System.out.printf(" %.2f ", value);
					}
				} catch (Exception e) {
					System.out.printf(" X ");
				}
				System.out.print("\t");

			}
			System.out.print("\n");
		}
		
		if(equal)
			System.out.println("============IDENTIQUES=============");
		
		return substract;
	}
	
	
	public Integer getHeightPx() {
		return heightPx;
	}

	public Integer getNodeCenterX() {
		return nodeCenterX;
	}

	public void setNodeCenterX(Integer nodeCenterX) {
		this.nodeCenterX = nodeCenterX;
	}

	public Integer getNodeCenterY() {
		return nodeCenterY;
	}

	public void setNodeCenterY(Integer nodeCenterY) {
		this.nodeCenterY = nodeCenterY;
	}

	public int getX(Node node) {
		return nodeIdToNodeXY(node.getNodeId())[0];
	}
	
	public int getY(Node node) {
		return nodeIdToNodeXY(node.getNodeId())[1];
	}


	public void setShowImage(Boolean showImage) {
		this.showImage = showImage;
	}
	
	public ImageNode getImageArea() {
		return imageArea;
	}

	public void setImageArea(ImageNode imageArea) {
		this.imageArea = imageArea;
	}



	
}
