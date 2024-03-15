package RN.dataset;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.RandomAccess;

public class OutputDataList <E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable{

	private List<Double[]> outputs; 
	
	public OutputDataList(Double[][] outputs){
		this.outputs = Arrays.asList(outputs);
	}
	
	public OutputDataList(){
		this.outputs = new ArrayList<Double[]>();
	}

	public void addData(Double[] value){
		outputs.add(value);
	}


	public E get(int index){
		return (E) new OutputData(outputs.get(index));
	}

	@Override
	public int size() {
		return outputs.size();
	}

	public void clear() {
		outputs.clear();
	}
	
}
