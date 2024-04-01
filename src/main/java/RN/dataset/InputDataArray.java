package RN.dataset;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.List;
import java.util.RandomAccess;

/**
 * @author Eric Marchand
 *
 */
public class InputDataArray<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2086895423766641269L;
	
	
	List<Double[]> inputs; 
	List<Double[]> ideals;
	
	
	public InputDataArray(Double[][] inputs, Double[][] ideals){
		this.inputs = Arrays.asList(inputs);
		this.ideals = Arrays.asList(ideals);
		
	}

	public List<Double[]> getInputs() {
		return inputs;
	}

	public void setInputs(List<Double[]> inputs) {
		this.inputs = inputs;
	}

	public List<Double[]> getIdeals() {
		return ideals;
	}

	public void setIdeals(List<Double[]> ideals) {
		this.ideals = ideals;
	}

	@Override
	public E get(int index) {
		return (E) new InputData(inputs.get(index), ideals.get(index));
	}

	@Override
	public int size() {
		return inputs.size();
	}

	

}
