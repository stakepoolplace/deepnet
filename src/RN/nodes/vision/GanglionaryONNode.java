package RN.nodes.vision;

import RN.algoactivations.EActivation;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.PixelNode;
import javafx.scene.layout.Pane;

public class GanglionaryONNode extends PixelNode {
	
	public GanglionaryONNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.GANGLIONARY_ON;
		//this.activationFx = EActivation.SYGMOID_0_1_INVERSE;
		this.activationFxPerNode = true;
	}

	public GanglionaryONNode() {
		super();
		this.nodeType = ENodeType.GANGLIONARY_ON;
		//this.activationFx = EActivation.SYGMOID_0_1_INVERSE;
		this.activationFxPerNode = true;
	}

	public GanglionaryONNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.GANGLIONARY_ON;
	}
	

	
	public void initParameters(int pixelCount){
		super.initParameters(pixelCount);
	}



	@Override
	public void newLearningCycle(int cycleCount) {
		// Randomisation des parametres à chaque cycle d'apprentissage

		// if(cycleCount % 100 == 0)
		// initRandomParameters();
	}


	
	@Override
	public void addGraphicInterface(Pane pane) {
		//((SecondDerivatedGaussianLinkage) getLinkage()).addGraphicInterface(pane);
		
	}


}
