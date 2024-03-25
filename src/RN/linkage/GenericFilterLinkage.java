package RN.linkage;

import java.util.List;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public class GenericFilterLinkage extends FilterLinkage {


	public GenericFilterLinkage() {
	}
	
	@Override
	public void initParameters() {
		

	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		SigmaWi sigmaWI = new SigmaWi();
		
		// somme des entrees pondérées
		ILayer sublayer = thisNode.getArea().getPreviousLayer();
		if(sublayer != null){
			
			IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			
			initFilter(this, ID_FILTER_GENERIC, (IPixelNode) thisNode, subArea);
			
			subArea.applyConvolutionFilter(this, ID_FILTER_GENERIC, (IPixelNode)thisNode, sigmaWI);
			
		}
		
		return sigmaWI.value();
	}

	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		initFilter(this, ID_FILTER_GENERIC, (IPixelNode) thisNode, subArea);
		
		subArea.applyConvolutionFilter(this, ID_FILTER_GENERIC, (IPixelNode) thisNode, 0.0000001f);

	}
	
	public void initFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, IAreaSquare subArea) {
		
		
		FilterIndex idx = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		
		if (this.getFilter(idx) == null) {
			
			Filter existingFilter = Filter.returnExistingFilter(eSampleFunction, params);
			Filter filter = null;
			if(existingFilter == null){
				
				filter = new Filter(idFilter, eSampleFunction, new Double[subArea.getWidthPx()][subArea.getHeightPx()], params);
				
				double filterValue = 0D;
				
				List<INode> nodeList = subArea.getNodes();
				INode sublayerNode = null;
				for (int index = 0; index < nodeList.size(); index++) {
					
					sublayerNode = nodeList.get(index);
					
					filterValue = linkage.processFilter(eSampleFunction, (IPixelNode) sublayerNode, params);

					// Ajout des valeurs discretes du filtre dans le cache matriciel
					if (Math.abs(filterValue) > 0.001D) {
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), filterValue);
					}else{
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), 0D);
					}
				}
				
				filter.resizeFilter();
				
				this.setFilter(idx,  filter);
				
			}else{
				
				this.setFilter(idx,  existingFilter);
				
			}

			this.getFilter(idx).filterToString();
		}
		
	}
	
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				(double) subArea.getHeightPx(), 
				(double) ((PixelNode) sublayerNode).getX(),
				(double) ((PixelNode) sublayerNode).getY(), 
				(double) ((PixelNode) subArea.getNodeCenterXY()).getX(), 
				(double) ((PixelNode) subArea.getNodeCenterXY()).getY());
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
					((IAreaSquare)thisArea).getFilter(ID_FILTER_GENERIC).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		
		pane.getChildren().addAll(hbox);
		
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



}
