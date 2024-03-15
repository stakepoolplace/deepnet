package RN.nodes.vision;

import RN.algoactivations.EActivation;
import RN.nodes.ENodeType;
import RN.nodes.INode;
import RN.nodes.PixelNode;
import javafx.scene.layout.Pane;

public class BiPolarLNode extends PixelNode {
	
	
	public BiPolarLNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.BIPOLAR_L;
		//this.activationFx = EActivation.SYGMOID_0_1_INVERSE;
		this.activationFxPerNode = true;
	}

	public BiPolarLNode() {
		super();
		this.nodeType = ENodeType.BIPOLAR_L;
		//this.activationFx = EActivation.SYGMOID_0_1_INVERSE;
		this.activationFxPerNode = true;
	}

	public BiPolarLNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.BIPOLAR_L;
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
