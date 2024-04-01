package RN.links;

/**
 * REGULAR, SHARED links : weights updated during backpropagation.
 * OTHERS : no update.
 * 
 * @author Eric
 *
 */
public enum ELinkType {

	REGULAR, SHARED, SELF_NODE_LINK, RECURRENT_LINK, RECURRENT_LATERAL_LINK, LAGGED_LINK, 
	
	// Vision 
	ON, OFF;
	
}
