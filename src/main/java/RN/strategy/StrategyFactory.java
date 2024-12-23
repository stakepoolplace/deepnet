package RN.strategy;

import RN.INetwork;
import RN.Network;

/**
 * @author Eric Marchand
 *
 */
public class StrategyFactory {

	public static Strategy create(INetwork network, EStrategy eStrategy){
		
		Strategy strategy = new Strategy(network);

		
		if(eStrategy == EStrategy.AUTO_GROWING_HIDDENS){
			strategy = new GrowingNetworkStrategy();
		}
		
		return strategy;
	}
	
	public static void main(String[] args){
		create(Network.getInstance(), EStrategy.AUTO_GROWING_HIDDENS).apply();
	}
	
}
