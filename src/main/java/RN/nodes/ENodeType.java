package RN.nodes;

/**
 * @author Eric Marchand
 *
 */
public enum ENodeType {

	//GAUSSIAN : used in vision to perform gaussian elliptic mask on a picture
	REGULAR("RN.nodes.Node"), 
	RECURRENT("RN.nodes.RecurrentNode"),
	TIMESERIE("RN.nodes.Node"), 
	ALL("RN.nodes.Node"), 
	LSTM("RN.nodes.LSTMNode"), 
	PIXEL("RN.nodes.PixelNode"),
	IMAGE("RN.nodes.ImageNode"),
	SIGMAPI("RN.nodes.SigmaPiNode"),
	PRODUCT("RN.nodes.ProductNode"),
	
	// Vision morphotypes
	BIPOLAR_L("RN.nodes.vision.BiPolarLNode"),
	BIPOLAR_S("RN.nodes.vision.BiPolarSNode"), 
	GANGLIONARY_OFF("RN.nodes.vision.GanglionaryOFFNode"),
	GANGLIONARY_ON("RN.nodes.vision.GanglionaryONNode");
	
	private String className;
	
	ENodeType(String className){
		this.setClassName(className);
	}

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}

	public static ENodeType[] arrayOf(String[] split) {
		ENodeType[] result = new ENodeType[split.length];
		int idx = 0;
		for(String value : split){
			result[idx++] = ENodeType.valueOf(value);
		}
		return result;
	}
	
}
