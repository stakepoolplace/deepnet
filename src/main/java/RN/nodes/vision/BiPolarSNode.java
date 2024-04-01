package RN.nodes.vision;

import RN.algoactivations.EActivation;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.PixelNode;
import javafx.scene.layout.Pane;

public class BiPolarSNode extends PixelNode {
	
	
	public BiPolarSNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.BIPOLAR_S;
		//this.activationFx = EActivation.SYGMOID_0_1;
		this.activationFxPerNode = true;
	}

	public BiPolarSNode() {
		super();
		this.nodeType = ENodeType.BIPOLAR_S;
		//this.activationFx = EActivation.SYGMOID_0_1;
		this.activationFxPerNode = true;
	}

	public BiPolarSNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.BIPOLAR_S;
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
