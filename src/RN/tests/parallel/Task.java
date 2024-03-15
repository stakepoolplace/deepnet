package RN.tests.parallel;

import java.util.concurrent.RecursiveAction;

public class Task extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3464928662682916133L;

	public Double result = 0D;
	private static final int THRESHOLD = 1;
	private PropagationProblem problem = null;

	public Task(PropagationProblem problem) {
		this.problem = problem;
	}

	@Override
	protected void compute() {
		
		if (problem.idxStop >= problem.layer.getNodeCount() - 1) { 

			return;
			
		} else {
			Task worker1 = new Task(new PropagationProblem( 0, problem.layer.getNodeCount() / 2));
			Task worker2 = new Task(new PropagationProblem(problem.layer.getNodeCount() / 2, problem.layer.getNodeCount() / 2 + 1));

			
//			for (int idx = 1; idx < problem.layer.getNodeCount(); idx += 10) {
//				worker2 = new Task(new PropagationProblem(problem.layer, idx, idx + 1));
//				result += worker2.compute();
//			}
			
			worker1.fork();
			worker2.compute();
			
			//result = worker2.compute() + worker1.join();


		}
	}

}
