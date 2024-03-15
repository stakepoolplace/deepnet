package RN.genetic.alteration;

import RN.IArea;
import RN.ILayer;
import RN.genetic.Genetic;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.Node;
import RN.nodes.RecurrentNode;
import RN.utils.StatUtils;

public class RecurrentAlteration extends AbstractAlterationDecorator implements IAlteration {

	
	private String geneticCodeAlteration = "";
	
	public RecurrentAlteration(Alteration alteration) {
		this.decoratedAlteration = alteration;
	}

	public RecurrentAlteration() {
	}

	@Override
	public void beforeProcess() {
		int count = 0;
		ILayer firstLayer = network.getFirstLayer();
		for (ILayer layer : network.getLayers()) {
			if (layer != firstLayer) {
					Link link = null;
					for (INode sourceNode : layer.getLayerNodes()) {
						if (StatUtils.randomize(10) == 1) {
							Node recurrent = new RecurrentNode(sourceNode);
							firstLayer.getArea(0).addNode(recurrent);
							link = Link.getInstance(ELinkType.REGULAR, false);
							link.setSourceNode(sourceNode);
							link.setWeightModifiable(false);
							recurrent.addInput(link);
							count++;
							System.out.println("recurrent node added. ");
						}
					}
				}
			}
		if(count > 0)
			geneticCodeAlteration += "+RN(" + count + ")" + Genetic.CODE_SEPARATOR;
	}

	@Override
	public void process() {
		int count = 0;
		for (ILayer layer : network.getLayers()) {
			for (IArea area : layer.getAreas()) {
				if (StatUtils.randomize(5) == 1) {
					System.out.println("recurrent nodes (if exists) lateral-linked. ");
					INode lastRecurrent = null;
					for (INode node : area.getNodes()) {
						if (node.getNodeType() == ENodeType.RECURRENT) {
							if (lastRecurrent != null){
								lastRecurrent.doubleLink(node, ELinkType.RECURRENT_LATERAL_LINK);
								count++;
							}
						}
					}
				}
			}
		}
		
		if(count > 0)
			geneticCodeAlteration += "-R-(" + count + ")" + Genetic.CODE_SEPARATOR;
	}

	@Override
	public void afterProcess() {


	}
	
	@Override
	public String geneticCodeAlteration(){
		
		network.setName(network.getName() + geneticCodeAlteration);
		
		return geneticCodeAlteration;
		
	}

}
