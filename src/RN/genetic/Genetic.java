package RN.genetic;

import RN.INetwork;



public class Genetic {

	// nb of generations
	private final static int MAX_LIFE_CYCLE = 4;
	
	public final static String CODE_SEPARATOR = ".";
	public final static String GENE_SEPARATOR = "||";
	
	
	public static INetwork process(){
		
		// At the beginning...
		Generation.createEnvironnement();
		
		// Adam and Eve...
		Generation generation = Generation.initialSeed();
		
		int lifeCycle = 0;
		
		do{
			// run network over samples
			generation.life();
			
			// choose the best network
			generation.naturalSelection();
			
			// generate networks with a new code alteration
			if(lifeCycle + 1 < MAX_LIFE_CYCLE)
				generation.replace(generation.seed());
		
		}while(++lifeCycle < MAX_LIFE_CYCLE);
		
		return generation.getWinner();
		
	}
	
	public static void main(String[] args){
		System.out.println(Genetic.process());
	}
	
	
}
