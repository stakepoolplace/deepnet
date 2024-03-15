package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

public class GaborLogLinkage extends FilterLinkage {

	//Number of orientations (between 3 to 20) 8 is a typical value
	private Double n_t = null;
	private Double t = null;
	
	//Number of scales of the multiresolution scheme
	private Double n_s = null;
	private Double s = null;
	
	


	public GaborLogLinkage() {
	}
	

	
	public void initParameters() {
		
		if(params.length != 4)
			throw new RuntimeException("Missing Gabor parameters'");
		
		
		n_t = params[0];
		n_s = params[1];
		t = params[2];
		s = params[3];
		
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_GABOR_LOG, ESamples.LOG_GABOR, 0.1D, (IPixelNode) thisNode, subArea, getN_t(), getN_s(), getT(), getS());
		
		subArea.applyConvolutionFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode) thisNode, sigmaWI);
			
		
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
			IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			
			initFilter(this, ID_FILTER_GABOR_LOG, ESamples.LOG_GABOR, 0.1D, (IPixelNode) thisNode, subArea, getN_t(), getN_s(), getT(), getS());
			
			subArea.applyConvolutionFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode)thisNode, 0.001f);
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre Log-Gabor
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				(double) sublayerNode.getX(), // - subArea.getWidthPx() / 2D) * 4D,
				(double) sublayerNode.getY(), // - subArea.getHeightPx() / 2D) * 4D,
				(double) subArea.getNodeCenterX(),
				(double) subArea.getNodeCenterY(),
				getN_s(),
				getS(),
				getN_t(),
				getT()
				);
		
		
	}
	
	

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT) {
		// TODO Auto-generated method stub
		
	}
	
	public void addGraphicInterface(Pane pane) {
		
		Button showFilterBtn = new Button("show filter");
		
		showFilterBtn.setTooltip(new Tooltip("show filter"));
		
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(showFilterBtn);
		
		showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

			@Override
			public void handle(ActionEvent event) {
				try {
					//FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_CONVOLUTION);
					((IAreaSquare)thisArea).getFilter(ID_FILTER_GABOR_LOG).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		
		pane.getChildren().addAll(hbox);
		
	}






	public Double getN_t() {
		return n_t;
	}



	public void setN_t(Double n_t) {
		this.n_t = n_t;
	}



	public Double getN_s() {
		return n_s;
	}



	public void setN_s(Double n_s) {
		this.n_s = n_s;
	}



	public Double getS() {
		return s;
	}



	public void setS(Double s) {
		this.s = s;
	}



	public Double getT() {
		return t;
	}



	public void setT(Double t) {
		this.t = t;
	}

}
