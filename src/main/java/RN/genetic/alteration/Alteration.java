package RN.genetic.alteration;

import RN.INetwork;
import RN.algotrainings.ITrainer;

/**
 * @author Eric Marchand
 *
 */
public class Alteration {

	protected static INetwork network = null;
	protected static ITrainer trainer = null;
	
	public Alteration(){
	}
	
	public Alteration(INetwork network, ITrainer trainer){
		Alteration.network = network;
		Alteration.trainer = trainer;
	}
	
	
	public void process(){
		System.out.println("Alteration process");
	}
	

	public void apply(){
		process();
	}
	
}
