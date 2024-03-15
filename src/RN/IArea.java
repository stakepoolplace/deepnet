package RN;

import java.util.List;

import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.ELinkage;
import RN.linkage.ELinkageBetweenAreas;
import RN.linkage.IFilterLinkage;
import RN.linkage.ILinkage;
import RN.linkage.SigmaWi;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;

public interface IArea {

	int getNodeCount();

	List<INode> getNodes();

	INode getNode(int index);

	List<INode> createNodes(int nodeCount);

	void addNode(INode node);

	ILinkage getLinkage();
	
	ILayer getLayer();
	
	Integer getAreaId();
	
	void setAreaId(int areaId);
	
	void initGraphics();
	
	Boolean isShowImage();
	
	void showImageArea();
	
	void removeNode(INode node);
	
	void removeNodes(List<INode> nodes);	
	
	ILayer getPreviousLayer();
	
	ILayer getNextLayer();

	IArea getLeftSibilingArea();
	
	Identification getIdentification();
	
	IArea configureNode(boolean bias, EActivation activation, ENodeType type);
		
	IArea configureNode(boolean bias, ENodeType... types);
	
	IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, Integer[] targetedAreas, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams);
	
	IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, Integer[] targetedAreas, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable,  Double... optParams);
	
	IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable,  Double... optParams);
	
	IArea configureLinkage(ELinkage linkage, ELinkageBetweenAreas linkageAreas, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams);
	
	IArea configureLinkage(ELinkage linkage, ESamples eSampleFunction, Integer sampling, boolean linkageWeightModifiable,  Double... optParams);
	
	IArea configureLinkage(ELinkage linkage, ESamples eSampleFunction, boolean linkageWeightModifiable, Double... optParams);
	
	int getNodeCountMinusRecurrentOnes();
	
	Double[] propagation(boolean playAgain, Double[] outputValues) throws Exception;
	
	void setLayer(ILayer layer);
	
	IArea deepCopy();
	
	void finalizeConnections();
	
	void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, Float ecartType);
	
	void applyFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, SigmaWi sigmaWI);
	
	IArea getRightSibilingArea();

	void prePropagation();
	
	void postPropagation();
	

}