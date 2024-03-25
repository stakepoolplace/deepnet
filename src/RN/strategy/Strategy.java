package RN.strategy;

import RN.INetwork;

/**
 * @author Eric Marchand
 *
 */
public class Strategy {

	protected static INetwork network = null;
	
	public Strategy(){
	}
	
	public Strategy(INetwork network){
		Strategy.network = network;
	}
	
	
	public void process(){
		System.out.println("Strategy process");
	}
	

	public void apply(){
		process();
	}
	
}
