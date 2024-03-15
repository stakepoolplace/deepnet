package RN.genetic.alteration;

import RN.INetwork;
import RN.Network;
import RN.algotrainings.BackPropagationTrainer;
import RN.algotrainings.ITrainer;

public class AlterationFactory {

	public static Alteration create(INetwork network, ITrainer trainer, EAlteration eAlteration){
		
		Alteration alteration = new Alteration(network, trainer);
//		Strategy strategy = new WeightUpdateStrategy();
//		strategy = new TestStrategy(strategy);
		
		if(eAlteration == EAlteration.GROWING_HIDDENS){
			alteration = new GrowingHiddensAlteration();
		}else if(eAlteration == EAlteration.RAND_LEARNING_RATE){
			alteration = new LearningRateAlteration();
		}else if(eAlteration == EAlteration.RAND_MOMENTUM){
			alteration = new MomentumAlteration();
		}else if(eAlteration == EAlteration.RECURRENCY_ADDING){
			alteration = new RecurrentAlteration();
		}
		
		return alteration;
	}
	
	public static void main(String[] args){
		create(Network.getInstance(), new BackPropagationTrainer(), EAlteration.GROWING_HIDDENS).apply();
	}
	
}
