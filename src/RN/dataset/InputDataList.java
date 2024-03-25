package RN.dataset;

import java.util.AbstractList;
import java.util.List;
import java.util.RandomAccess;

/**
 * @author Eric Marchand
 *
 */
public class InputDataList<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2086895423766641269L;
	
	
	private List<List> inputs; 
	private List<List> ideals;
	
	
	public InputDataList(List<List> inputs, List<List> ideals){
		this.inputs = inputs;
		this.ideals = ideals;
		
	}



	@Override
	public E get(int index) {
		
		return (E) new InputData(inputs.get(index), ideals == null ? null : ideals.get(index));
	}

	@Override
	public int size() {
		return inputs.size();
	}



	public List<List> getInputs() {
		return inputs;
	}



	public void setInputs(List<List> inputs) {
		this.inputs = inputs;
	}



	public List<List> getIdeals() {
		return ideals;
	}



	public void setIdeals(List<List> ideals) {
		this.ideals = ideals;
	}



	public void clear() {
		inputs.clear();
		ideals.clear();
	}

	

}
