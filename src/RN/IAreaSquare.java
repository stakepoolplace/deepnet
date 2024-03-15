package RN;

import java.util.List;

import RN.dataset.inputsamples.ESamples;
import RN.linkage.Filter;
import RN.linkage.IFilterLinkage;
import RN.linkage.ILinkage;
import RN.linkage.SigmaWi;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.ImageNode;
import RN.nodes.Node;
import RN.nodes.PixelNode;
import javafx.scene.paint.Color;

public interface IAreaSquare {

	int getNodeCount();
	
	PixelNode getNode(int index);
	
	List<INode> getNodes();
	
	ILinkage getLinkage();
	
	IPixelNode getNodeXY(int x, int y);
	
	Integer nodeXYToNodeId(int x, int y, int sampling);
	
	IPixelNode getNodeXY(int x, int y, int sampling);

	IPixelNode getNodeXY(int x, int y, int x0, int y0, double theta);

	List<IPixelNode> getNodesInSquareZone(int x0, int y0, int width, int height);

	List<IPixelNode> getNodesInCirclarZone(int x0, int y0, int radius);
	
	List<IPixelNode> getNodesOnCirclarPerimeter(int x0, int y0, int radius);

	IPixelNode getNodeCenterXY();
	
	int getX(Node node);
	
	int getY(Node node);
	
	Integer getWidthPx();
	
	Integer getHeightPx();
	
	ImageNode getImageArea();
	
	Integer getNodeCenterX();
	
	Integer getNodeCenterY();
	
	Integer nodeXYToNodeId(int x, int y);
	
	int[] nodeIdToNodeXY(int id);
	
	Boolean isShowImage();
	
	void showImageArea();
	
	Identification getIdentification();
	
	void applyConvolutionFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, Float ecartType);
	
	void applyConvolutionCompositeFilter(IFilterLinkage linkage, int idFilter1, int idFilter2, ESamples op, IPixelNode thisNode, Float ecartType);
	
	void applyConvolutionCompositeFilter(IFilterLinkage linkage, int idFilter1, int idFilter2, ESamples op, IPixelNode thisNode, SigmaWi sigmaWI);
	
	void applyConvolutionFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, SigmaWi sigmaWI);
	
	void applyMaxPoolingFilter(IFilterLinkage linkage, int width, int stride, IPixelNode thisNode);
	
	void applyMaxPoolingFilter(IFilterLinkage linkage, int width, int stride, IPixelNode thisNode, SigmaWi sigmaWI);
	
	void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, Float ecartType);
	
	void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, SigmaWi sigmaWI);
	
	IArea getLeftSibilingArea();
	
	void pixelsToString(List<IPixelNode> pixels);
	
	void showGradients(double magnitudeFactor, double magnitudeThreshold, int sampling, Color color);
	
	void setShowImage(Boolean showImage);
	
	Filter getFilter(int filterId);
	
	

}