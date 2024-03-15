package RN.linkage;

import java.util.List;
import java.util.Map;

import RN.IArea;
import RN.ILayer;
import RN.Identification;
import RN.links.Link;
import RN.links.Weight;
import RN.nodes.INode;
import javafx.scene.layout.Pane;

public interface ILinkage {
	
	
	void sublayerFanOutLinkage(INode thisNode, ILayer sublayer);

	void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT);

	void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer);

	void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT);

	void setWeightModifiable(boolean weightModifiable);
	
	Boolean isWeightModifiable();

	void setParams(Double[] params);
	
	Double[] getParams();

	void initParameters();
	
	void addGraphicInterface(Pane pane);
	
	void setLinkageType(ELinkage linkageType);
	
	ELinkage getLinkageType();

	double getSigmaPotentials(INode thisNode);
	
	double getLinkedSigmaPotentials(INode thisNode);
	
	double getUnLinkedSigmaPotentials(INode thisNode);

	IArea getArea();

	void setArea(IArea thisArea);

	void setSampling(Integer sampling);
	
	Integer getSampling();
	
	void setLinkageAreas(ELinkageBetweenAreas linkageAreas);

	void setTargetedAreas(List<Integer> targetedArea);
	
	List<IArea> getLinkedAreas();

	IArea getLinkedArea();

	void prePropagation();
	
	void postPropagation();
	
	//Area initAreaSubSampling(Area subArea);
	

}