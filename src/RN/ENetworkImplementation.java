package RN;

/**
 * @author Eric Marchand
 * 
 */
public enum ENetworkImplementation {
	
	LINKED("Link objects' are shared between nodes."), UNLINKED("Computational is done without link object.");
	
	String comment = null;
	ENetworkImplementation(String comment){
		this.comment = comment;
	}

}
