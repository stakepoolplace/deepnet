package RN;

import java.io.Serializable;

/**
 * @author Eric Marchand
 * 
 */
public class NetworkElement implements Serializable{

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
