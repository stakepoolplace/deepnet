package RN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import RN.dataset.InputData;
import RN.dataset.InputDataList;

/**
 * @author Eric Marchand
 * 
 */
public class DataSeries {

	private static DataSeries instance = null;
	
	private List<List> INPUTS = new ArrayList<List>();
	private List<List> INPUTTESTS = new ArrayList<List>();
	private List<List> IDEALS = new ArrayList<List>();
	
	private List<InputData> inputDataSet = new InputDataList<InputData>(INPUTS, IDEALS);
	private List<InputData> inputTestDataSet = new InputDataList<InputData>(INPUTTESTS, null);
	
	public List<InputData> getInputTestDataSet() {
		return inputTestDataSet;
	}


	public static DataSeries getInstance(){
		if(instance == null){
			instance = new DataSeries();
		}
		
		return instance;
	}
	

	public List<InputData> getInputDataSet() {
		return inputDataSet;
	}

	public List<List> getINPUTTESTS() {
		return INPUTTESTS;
	}


	public void setINPUTTESTS(List<List> iNPUTTESTS) {
		INPUTTESTS = iNPUTTESTS;
	}	
	

	public void clearTests() {
		INPUTTESTS.clear();
	}


	public boolean testsAreEmpty() {
		return INPUTTESTS.isEmpty();
	}


	public List<List> getINPUTS() {
		return INPUTS;
	}

	public void setINPUTS(List<List> iNPUTS) {
		INPUTS = iNPUTS;
	}

	public List<List> getIDEALS() {
		return IDEALS;
	}

	public void setIDEALS(List<List> iDEALS) {
		IDEALS = iDEALS;
	}
	

	
	public void addINPUTS(List<Double> inputs) {
		getINPUTS().add(inputs);
	}

	public void addIDEALS(List<Double> ideals) {
		getIDEALS().add(ideals);
	}
	
	public boolean addTests(List<Double> arg0) {
		return getINPUTTESTS().add(arg0);
	}
	

	public void clearSeries() {
		getINPUTS().clear();
		getIDEALS().clear();
		getINPUTTESTS().clear();
		inputDataSet.clear();
	}
	
	public void addINPUT(Double... inputs) {
		getINPUTS().add(Arrays.asList(inputs));
	}

	public void addIDEAL(Double... ideals) {
		getIDEALS().add(Arrays.asList(ideals));
	}


	public String getString() {
		String result = "";
		result = "INPUTS:" + INPUTS.size()  + " IDEALS:" + IDEALS.size() + "\r\n";
		int idxV = 1;
		for(List lineInput : INPUTS){
			Iterator itr = lineInput.iterator();
			result += idxV++ ;
			while(itr.hasNext()){
				Double value = (Double) itr.next();
				result += " V:" + value + " ";
			}
			result += "\r\n";
			result += lineInput.size() + " inputs \r\n";
			result += "\r\n";
		}
		int idxI = 1;
		for(List lineInput : IDEALS){
			Iterator itr = lineInput.iterator();
			result += idxI++ ;
			while(itr.hasNext()){
				Double value = (Double) itr.next();
				result += "I:" + value + " ";
			}
			result += "\r\n";
			result += lineInput.size() + " ideals \r\n";
			result += "\r\n";
		}
		return result;
	}
	


	
}
