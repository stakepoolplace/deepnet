package RN;

/**
 * @author Eric Marchand
 * 
 */
public class NetworkElement {

	protected static NetworkContext context = NetworkContext.getContext();
	
	protected static Network network = null;
	

	public static NetworkContext getContext() {
		return context;
	}


	public static void initContext() {
		context = new NetworkContext();
		NetworkContext.setContext(context);
	}


}
