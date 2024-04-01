package RN.algoactivations;

/**
 * Interface IPerformer Perform a task for statistics Used in AdapterStatistics
 * 
 * @author emarchand
 * 
 */
public interface IActivation {
   
   double perform(double... value) throws Exception;
   
   double performDerivative(double... value) throws Exception;
   
}
