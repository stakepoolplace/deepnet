package RN.fxml.controllers;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import RN.TestNetwork;
import RN.algotrainings.LSTMTrainer;

public class NN implements Initializable {

	@FXML
	public static TextField valinp0;
	public static TextField valinp1;
	public static TextField valinp0out;
	public static TextField valinp1out;
	public static TextField valhid0inp;
	public static TextField valhid1inp;
	public static TextField valhid2inp;
	public static TextField valhid0out;
	public static TextField valhid1out;
	public static TextField valhid2out;
	public static TextField valout0inp;
	public static TextField valout0out;
	public static TextField errhid0;
	public static TextField errhid1;
	public static TextField errhid2;
	public static TextField errout0;
	public static TextField validealout0;
	public static TextField valaggout0;
	public static TextField valagghid0;
	public static TextField valagghid1;
	public static TextField valagghid2;
	public static Label whid0inp0;
	public static Label whid0inp1;
	public static Label whid1inp0;
	public static Label whid1inp1;
	public static Label whid2inp0;
	public static Label whid2inp1;	
	public static Label wbiashid0;	
	public static Label wbiashid1;	
	public static Label wbiashid2;	
	public static Label wbiasout0;	
	public static Label wout0hid0;	
	public static Label wout0hid1;	
	public static Label wout0hid2;	
	
	
	
	@FXML
	@Override
	public void initialize(URL arg0, ResourceBundle arg1) {
		
	}
	
	@FXML
	public static void show(){
		System.out.println("clicked");
		TestNetwork.getInstance().getNetwork().show();

	}
	
	public static void selectTrainingSet(){
		System.out.println("selectTrainingSet clicked");
	}
	
	public static void nextTrainInputValues(){
		
		LSTMTrainer.getInstance().nextTrainInputValues();

		System.out.println("nextTrainInputValues clicked");
		TestNetwork.getInstance().getNetwork().show();
	}
	
	
	public void feedForward() throws Exception{
		
		LSTMTrainer.getInstance().feedForward();
		System.out.println("feedForward clicked");
		TestNetwork.getInstance().getNetwork().show();
	}
	
	public void computeDeltaWeights() throws Exception{
		
		LSTMTrainer.getInstance().computeDeltaWeights();
		System.out.println("computeDeltaWeights clicked");
		TestNetwork.getInstance().getNetwork().show();
	}
	
	public void updateAllWeights() throws Exception {
		
		LSTMTrainer.getInstance().updateAllWeights();
		System.out.println("updateAllWeights clicked");
		TestNetwork.getInstance().getNetwork().show();
	}
	
	
	public void doBatch() throws Exception{
//		selectTrainingSet();
//		for(int ind=1; ind <= SimpleTraining.getInstance().getInputDataSet().size(); ind++){
			nextTrainInputValues();
			feedForward();
			computeDeltaWeights();
//		}
//		updateAllWeights();
	}
	
	public void doUpdateAfterEachTrainings() throws Exception{
//		selectTrainingSet();
//		for(int ind=1; ind <= SimpleTraining.getInstance().getInputDataSet().size(); ind++){
			nextTrainInputValues();
			feedForward();
			computeDeltaWeights();
			updateAllWeights();
//		}
		
	}
	
	public void initWeights(){
		TestNetwork.getInstance().initWeights(-1.0D, 1.0D);
		System.out.println("initWeights clicked");
		TestNetwork.getInstance().getNetwork().show();
	}
	
	
	


}
