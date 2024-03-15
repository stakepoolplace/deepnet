package RN.genetic;

import java.util.Arrays;
import java.util.Comparator;

import RN.INetwork;
import RN.ITester;
import RN.Network;
import RN.TestNetwork;
import RN.ViewerFX;
import RN.algotrainings.BackPropagationTrainer;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.InputSample;
import RN.genetic.alteration.AlterationFactory;
import RN.genetic.alteration.EAlteration;
import RN.utils.StatUtils;

public class Generation {

	
	public static ITester tester = null;
	
	private INetwork[] generation;
	private ITrainer[] trainers;
	private static final int GENERATION_SIZE = 20;
	private static int generationCount = 0;
	
	private INetwork winner = null;
	

	public INetwork getWinner() {
		return winner;
	}


	public void setWinner(INetwork winner) {
		this.winner = winner;
	}


	public Generation(INetwork[] generation, ITrainer[] trainers) {
		this.generation = generation;
		this.trainers = trainers;
	}
	
	
	public void replace(INetwork[] futureGeneration){
		this.generation = futureGeneration;
	}
	
	/**
	 * At the beginning, God created the heaven and the earth...
	 */
	public static void createEnvironnement(){
		tester = TestNetwork.getInstance();
//		ViewerFX.setTrainer(trainer);
//		ViewerFX.setTester(tester);
//		ViewerFX.startViewerFX();
	}
	
	/**
	 * Adam and Eve, equals but sexually different...
	 * 
	 * @return
	 */
	public static Generation initialSeed(){
		
		try {
			InputSample.setFileSample(tester, tester.getFilePath(), 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		INetwork god = tester.getNetwork();
		
		Generation initialGeneration = new Generation(new INetwork[GENERATION_SIZE], new ITrainer[GENERATION_SIZE]);
		
		initialGeneration.generation[0] = god.deepCopy(generationCount);
		initialGeneration.trainers[0] = new BackPropagationTrainer();
		
		for(int idx=1; idx < GENERATION_SIZE; idx++){
			initialGeneration.trainers[idx] = new BackPropagationTrainer();
			initialGeneration.generation[idx] = codeAlteration(god.deepCopy(generationCount), initialGeneration.trainers[idx]);
		}
		
		generationCount++;
		
		return initialGeneration;
		
	}
	
	public void life() {
		
		int maxTrainingCycles = 100;
		
		for(int idx=0; idx < generation.length; idx++){
			
			tester.setNetwork(generation[idx]);
			trainers[idx].setMaxTrainingCycles(maxTrainingCycles);
			
			try {
				trainers[idx].launchTrain();
				System.out.println("life " + idx + " of " + generation.length + " complete.");
			} catch (Exception e) {
				// dead during training, god bless you !
				generation[idx] = null;
				trainers[idx] = null;
				System.out.println("dead during training, god bless you !");
			}
			
		}
		
	}
	
	public void naturalSelection(){
		
		Comparator<INetwork> sortByError = new SortByBackPropagationError();
		Arrays.sort(generation, sortByError);
		setWinner(generation[0]);
		
		int i = 1;
		for(INetwork net : generation){
			System.out.print("Results for natural selection  >>     "+ "   (error : " + StatUtils.format(net.getAbsoluteError(), "###.#####")+")         " + ( i == 0 ? "WINNER" : i) + " : " + net );
			System.out.println("       =  " + net.geneticCodec() );
			i++;
		}
		
		
	}
	
	public INetwork[] seed(){
		
		INetwork winner = getWinner();
		ITrainer trainer = null;
		System.out.println("START Replications of the winner. [x" + GENERATION_SIZE + "]");
		INetwork[] futureGeneration = new INetwork[GENERATION_SIZE];
		if(!winner.getName().endsWith(Genetic.GENE_SEPARATOR))
			winner.appendName(Genetic.GENE_SEPARATOR);
		futureGeneration[0] = winner.deepCopy(generationCount);
		trainers[0] = new BackPropagationTrainer();
		for(int idx=1; idx < GENERATION_SIZE; idx++){
			System.out.println("Replication with alteration of " + winner + "]");
			trainer = new BackPropagationTrainer();
			trainers[idx] = trainer;
			futureGeneration[idx] = codeAlteration(winner.deepCopy(generationCount), trainer);
			
		}
		generationCount++;
		
		System.out.println("END of Replications---------------- ");
		return futureGeneration;
		
	}
	
	public static INetwork codeAlteration(INetwork network, ITrainer trainer){
		// Math random EAlteration
		for(int nbAlt = 0; nbAlt < StatUtils.randomize(50); nbAlt++){
			System.out.println("Alteration for " + network + " :");
			int altKind = StatUtils.randomize(3);
			switch(altKind){
				case 0 : 
					System.out.println("growing hiddens alteration");
					AlterationFactory.create(network, trainer, EAlteration.GROWING_HIDDENS).apply();
					break;
				case 1 : 
					System.out.println("learning rate alteration");
					AlterationFactory.create(network, trainer, EAlteration.RAND_LEARNING_RATE).apply();
					break;
				case 2 : 
					System.out.println("momentum alteration");
					AlterationFactory.create(network, trainer, EAlteration.RAND_MOMENTUM).apply();
					break;
				case 3 : 
					System.out.println("recurrency alteration");
					AlterationFactory.create(network, trainer, EAlteration.RECURRENCY_ADDING).apply();
					break;					
				default :
					System.out.println("no alteration");
			}
		}
		
		return network;
	}
	
	
	
	public static void main(String[] args){
		for(int nbAlt = 0; nbAlt < StatUtils.randomize(100); nbAlt++){
			System.out.println("nbAlt =" + nbAlt);
			int altKind = StatUtils.randomize(1);
			System.out.println("kind : " + altKind);
			switch(altKind){
				case 0 : 
					System.out.println("growing hiddens alteration");
					break;
				default :
					System.out.println("no alteration");
			}
		}
	}
	
}
