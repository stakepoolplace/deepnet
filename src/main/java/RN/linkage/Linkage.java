package RN.linkage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import RN.ENetworkImplementation;
import RN.IArea;
import RN.ILayer;
import RN.Identification;
import RN.NetworkElement;
import RN.links.ELinkType;
import RN.links.Link;
import RN.links.Weight;
import RN.nodes.INode;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public abstract class Linkage extends NetworkElement implements ILinkage {
	
	protected ELinkage linkageType = null;
	protected ELinkageBetweenAreas linkageAreas = ELinkageBetweenAreas.ONE_TO_MANY;
	protected List<Integer> targetedAreas = null;
	Boolean weightModifiable = Boolean.FALSE;
	protected Double[] params = null;
	protected IArea thisArea = null;
	protected Integer sampling = 1;
	
	// When we have no link, weights are stored here.
	protected static Map<Identification, Map<Identification, Link>> linksByTargetNode = new HashMap<Identification, Map<Identification, Link>>();
	protected static Map<Identification, Map<Identification, Link>> linksBySourceNode = new HashMap<Identification, Map<Identification, Link>>();
	
	
	@Override
	public abstract void sublayerFanOutLinkage(INode thisNode, ILayer sublayer);

	@Override
	public abstract void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT);

	@Override
	public abstract void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer);

	@Override
	public abstract void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT);
	

	
	public List<IArea> getLinkedAreas(){
		
		List<IArea> list = new ArrayList<IArea>();
		
		if(linkageAreas == ELinkageBetweenAreas.MANY_TO_ONE){
			
			if(targetedAreas == null || targetedAreas.isEmpty()){
				return thisArea.getPreviousLayer().getAreas();
			}else{
				
				for(IArea area : thisArea.getPreviousLayer().getAreas()){
					for(Integer id : targetedAreas){
						if(area.getAreaId() == id)
							list.add(area);
					}
				}
				
				return list;
				
			}
			
		}else if(linkageAreas == ELinkageBetweenAreas.ONE_TO_MANY){
			
			list.add(thisArea.getPreviousLayer().getArea(targetedAreas == null ? 0 : targetedAreas.get(0)));
			
		}else if(linkageAreas == ELinkageBetweenAreas.ONE_TO_ONE){
			
			list.add(thisArea.getPreviousLayer().getArea(thisArea.getAreaId()));
		}
		
		return list;
	}
	

	
	public IArea getLinkedArea(){
		
		if(thisArea.getLayer().isFirstLayer())
			return null;
		
		if(linkageAreas == ELinkageBetweenAreas.MANY_TO_ONE){
			
			return thisArea.getPreviousLayer().getAreas().get(0);
			
		}else if(linkageAreas == ELinkageBetweenAreas.ONE_TO_MANY){
			
			return thisArea.getPreviousLayer().getArea(targetedAreas == null ? 0 : targetedAreas.get(0));
			
		}else if(linkageAreas == ELinkageBetweenAreas.ONE_TO_ONE){
			
			if(targetedAreas == null){
				return thisArea.getLeftSibilingArea();
			}else{
				return thisArea.getPreviousLayer().getArea(targetedAreas.get(0));
			}
			
		}
		
		if(targetedAreas != null){
			return thisArea.getPreviousLayer().getArea(targetedAreas.get(0));
		}
		
		
		return null;
	}
	
	
	public double getLinkedSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		// somme des entrees pondérées
		for (Link input : thisNode.getInputs()) {
			if(getContext().getClock() == -1 || input.getFireTimeT() == getContext().getClock()){
				
				sigmaWI += input.getValue() * input.getWeight();
				
				input.synchFutureFire();
			}
			
		}
		
		// ajout du biais
		if (thisNode.getBiasInput() != null){
			if(getContext().getClock() == -1 || thisNode.getBiasInput().getFireTimeT() == getContext().getClock()){
				
				sigmaWI -= thisNode.getBiasInput().getValue() * thisNode.getBiasInput().getWeight();
				
				thisNode.getBiasInput().synchFutureFire();
			}
		}
		
		return sigmaWI;
	}
	
	public double getSigmaPotentials(INode thisNode){
		
		Double sigmaWI = 0D;
		
		if(network != null && network.getImpl() == ENetworkImplementation.UNLINKED){
			
			sigmaWI = getUnLinkedSigmaPotentials(thisNode);
			
		}else{
			
			sigmaWI = getLinkedSigmaPotentials(thisNode);
			
		}
		
		return sigmaWI;
		
	}
	
	
	public void addGraphicInterface(Pane pane){
	}
	
	public void prePropagation(){
	}
	
	public void postPropagation(){
	}
	

	public Boolean isWeightModifiable() {
		return weightModifiable;
	}

	@Override
	public void setWeightModifiable(boolean weightModifiable) {
		this.weightModifiable = weightModifiable;
	}
	
	@Override
	public void setParams(Double[] params){
		this.params = params;
	}
	
	@Override
	public Double[] getParams(){
		return this.params;
	}

	public ELinkage getLinkageType() {
		return linkageType;
	}

	public void setLinkageType(ELinkage linkageType) {
		this.linkageType = linkageType;
	}

	public IArea getArea() {
		return thisArea;
	}

	public void setArea(IArea thisArea) {
		this.thisArea = thisArea;
	}
	
	public void setSampling(Integer sampling){
		this.sampling = sampling;
	}

	public Integer getSampling() {
		return sampling;
	}

	public ELinkageBetweenAreas getLinkageAreas() {
		return linkageAreas;
	}

	public void setLinkageAreas(ELinkageBetweenAreas linkageAreas) {
		this.linkageAreas = linkageAreas;
	}

	public List<Integer> getTargetedAreas() {
		return targetedAreas;
	}

	public void setTargetedAreas(List<Integer> targetedArea) {
		this.targetedAreas = targetedArea;
	}
	
	public static Map<Identification, Map<Identification, Link>> getLinks() {
		return linksByTargetNode;
	}
	
	public static Map<Identification, Link> getInputLinks(Identification targetNodeId) {
		
		Map<Identification, Link> results = linksByTargetNode.get(targetNodeId);
		
		if(results == null){
			return new HashMap<Identification, Link>();
		}
		
		return results;
		
	}
	
	public Link getLink(Identification targetNodeId, Identification sourceNodeId, Link defaultWeight) {
		
		Link weight = linksByTargetNode.get(targetNodeId).get(sourceNodeId);
		
		if(weight == null)
			return defaultWeight;
		
		return weight;
	}

	private static void putLink(Identification targetNode, Identification sourceNode, Link link) {
		
		Map<Identification, Link> linksByTarget = linksByTargetNode.putIfAbsent(targetNode, new HashMap<Identification, Link>());
		if(linksByTarget != null){
			linksByTarget.put(sourceNode, link);
		}else{
			linksByTargetNode.get(targetNode).put(sourceNode, link);
		}
		
		Map<Identification, Link> linksBySource = linksBySourceNode.putIfAbsent(sourceNode, new HashMap<Identification, Link>());
		if(linksBySource != null){
			linksBySource.put(targetNode, link);
		}else{
			linksBySourceNode.get(sourceNode).put(targetNode, link);
		}
		
	}
	
	public static Link getLinkAndPutIfAbsent(INode targetNode, INode sourceNode, boolean isWeightModifiable) {
		return getLinkAndPutIfAbsent( targetNode, sourceNode, isWeightModifiable, null);
	}
	
	public static Link getLinkAndPutIfAbsent(INode targetNode, INode sourceNode, boolean isWeightModifiable, Weight weight) {
		
		Link link = null;
		Link defaultLink = null;
		
		try{
			
			link = linksByTargetNode.get(targetNode.getIdentification()).get(sourceNode.getIdentification());
			
		}catch(Throwable e){
			
			if(weight == null)
				defaultLink = Link.getInstance(ELinkType.REGULAR, isWeightModifiable);
			else
				defaultLink = Link.getInstance(weight, isWeightModifiable);
			
			defaultLink.setTargetNode(targetNode);
			defaultLink.setSourceNode(sourceNode);
			
			putLink(targetNode.getIdentification(), sourceNode.getIdentification(), defaultLink);
			link = defaultLink;
			
		}finally{
			
			if(link == null){
				
				if(weight == null)
					defaultLink = Link.getInstance(ELinkType.REGULAR, isWeightModifiable);
				else
					defaultLink = Link.getInstance(weight, isWeightModifiable);
				
				defaultLink.setTargetNode(targetNode);
				defaultLink.setSourceNode(sourceNode);
				
				putLink(targetNode.getIdentification(), sourceNode.getIdentification(), defaultLink);
				link = defaultLink;
			}
			
		}
		
		return link;
	}
	
	public static Map<Identification, Map<Identification, Link>> getLinksBySourceNode() {
		return linksBySourceNode;
	}

	public static void setLinksBySourceNode(Map<Identification, Map<Identification, Link>> linksBySource) {
		linksBySourceNode = linksBySource;
	}

	


}
