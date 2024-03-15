package RN.linkage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.Coordinate;
import RN.dataset.inputsamples.ESamples;
import RN.dataset.inputsamples.InputSample;
import RN.links.ELinkType;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;
import RN.utils.MathUtils;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

public class LogPolarLinkage extends FilterLinkage {
	
	private Slider numOrientsSlider = null;
	private Slider numScalesSlider = null;
	
	// cache radius by angle of concentric circles zones.
	private static Map<Integer, List<Coordinate>> retina = new HashMap<Integer, List<Coordinate>>();

	//Number of orientations (between 3 to 20) 8 is a typical value
	private Double n_t = null;
	
	//Number of scales of the multiresolution scheme
	private Double n_s = null;
	
	private Double base = null;
	
	private static boolean cleaningDone = false;
	


	public LogPolarLinkage() {
		cleaningDone = false;
	}
	

	
	public void initParameters() {
		
		if(params.length == 0)
			throw new RuntimeException("Missing Log Polar parameters' (theta)");
		
		
		n_t = params[0];
		
		if(params.length == 2)
			n_s = params[1];
		
		base = 1 + (Math.PI / (Math.sqrt(3D) * n_t));
		
		cleaningDone = false;
		
	}
	
	public void postPropagation(){
		
		if(!cleaningDone){
			System.out.println("Begin cleaning... Node's count is : " + getArea().getNodeCount());
			
			IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			List<INode> nodesToRemove = new ArrayList<INode>();
			Coordinate coord = null;
			for(INode node : getArea().getNodes()){
				
				coord = isAtCenter(subArea, (PixelNode) node);
				
				if(coord == null){
					nodesToRemove.add((PixelNode) node);
				}
				
			}
			
			INode nodeToRemove = null;
			for(int idx=0; idx < nodesToRemove.size(); idx++){
				nodeToRemove = nodesToRemove.get(idx);
				getArea().removeNode(nodeToRemove);
			}
			
			nodesToRemove = null;
			
			
			System.out.println("Layer cleared. (Pixels deleted) Node's count is now : " + getArea().getNodeCount());
			
			cleaningDone = true;
		}
		
	}



	
	private void concentricCircleCenters(IAreaSquare area, Double angleCount, Double scaleCount) {
		
		
		double arc = 2D * Math.PI / angleCount;
		double p_fovea = angleCount / (2D * Math.PI);
		
		Double p_r = null;
		
		//double t_radius = arc * 1/3;
		
		double theta = 0D;
		List<Coordinate> ys = null;
		Coordinate coord = null;
		Integer idScale = 0;
		/*for(int idScale=0; idScale <= scaleCount; idScale++)*/
		do{
			
			for(int idAngle=0; idAngle <= angleCount; idAngle++){
				
				theta = arc * (idAngle + MathUtils.odd(idScale) / 2D);
				p_r = p_fovea * Math.pow(this.base, idScale);
				
				coord = new Coordinate();
				coord.setBase(this.base);
				coord.setTheta(theta);
				coord.setP(p_r);
				coord.setX0((double) ((IAreaSquare) getArea()).getNodeCenterX());
				coord.setY0((double) ((IAreaSquare) getArea()).getNodeCenterY());
				coord.logPolarToLinearSystem();
				
				if(retina.containsKey(coord.getX().intValue())){
					retina.get(coord.getX().intValue()).add(coord);
				}else{
					ys = new ArrayList<Coordinate>();
					ys.add(coord);
					retina.put(coord.getX().intValue(), ys);
				}
				
			}
			
			idScale++;
			
			if(scaleCount != null && idScale == scaleCount.intValue())
				break;
			
		}while( (Math.abs(coord.getX()) <= area.getWidthPx()) && Math.abs(coord.getY()) <= area.getHeightPx());
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		Double t_radius = null;
		Coordinate coord = isAtCenter(subArea, pix);
		
		if(coord != null){
			
			t_radius = (2D * Math.PI * coord.getP()) / (3D * n_t);
			
//			for(IPixelNode innerPix : pix.getAreaSquare().getNodesOnCirclarPerimeter(pix.getX(), pix.getY(), t_radius.intValue())){
//				((INode)innerPix).setComputedOutput(1D);
//			}
			
			// Champ récépteur
			List<IPixelNode> pixels = subArea.getNodesInCirclarZone(pix.getX(), pix.getY(), t_radius.intValue());
			System.out.println(pix + " : " + pixels.size() + " pixels.");
			for(IPixelNode innerPix : pixels){
				// Cellule ON/OFF center exitateur, pourtour inihibiteur
				if(innerPix.distance(pix) <= t_radius / 2D)
					sigmaWI.sum(innerPix.getComputedOutput());
				else
					sigmaWI.sum(-innerPix.getComputedOutput());
			}
			
		}
		
		//return thisNode.getComputedOutput();
		return sigmaWI.value();
	}
	
	private Coordinate isAtCenter(IAreaSquare area, IPixelNode pix) {
		
		if(retina.isEmpty()){
			concentricCircleCenters(area, n_t, n_s);
		}
		
		List<Coordinate> ysByXs = retina.get(pix.getX());
		
		if(ysByXs != null){
			for(Coordinate coord : ysByXs){
				if(coord != null && Math.abs(coord.getY() - pix.getY()) < 0.5D)
					return coord;
			}
		}
		
		return null;
	}



	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		IPixelNode pix = (IPixelNode) thisNode;
		
		Double t_radius = null;
		Coordinate coord = isAtCenter(subArea, pix);
		
		if(coord != null){
			
			t_radius = (2D * Math.PI * coord.getP()) / (3D * n_t);
			
//			for(IPixelNode innerPix : pix.getAreaSquare().getNodesOnCirclarPerimeter(pix.getX(), pix.getY(), t_radius.intValue())){
//				((INode)innerPix).setComputedOutput(1D);
//			}
			
			// Champ récépteur
			for(IPixelNode innerPix : subArea.getNodesInCirclarZone(pix.getX(), pix.getY(), t_radius.intValue())){
				// Cellule ON/OFF center exitateur, pourtour inihibiteur
				if(innerPix.distance(pix) <= t_radius / 2D)
					innerPix.link(thisNode, ELinkType.REGULAR, isWeightModifiable(), 1D);
				else
					innerPix.link(thisNode, ELinkType.REGULAR, isWeightModifiable(), -1D);
			}
			
		}
		
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		// Calcul du filtre Log-Gabor
		return  InputSample.getInstance().compute(
				filterFunction, 
				(double) subArea.getWidthPx(),
				getN_t(),
				getN_s(),
				(double) (sublayerNode.getX() - subArea.getWidthPx() / 2D) * 4D,
				(double) (sublayerNode.getY() - subArea.getHeightPx() / 2D) * 4D
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
		
		TextField logTF = new TextField("");
		
		numOrientsSlider = new Slider(0D, 360D, n_t);
		numOrientsSlider.setBlockIncrement(1D);
		
		double numScales = 0D;
		if(n_s == null)
			numScales = 0D;
		else
			numScales = n_s;
		
		numScalesSlider = new Slider(0D, 8D, numScales);
		numScalesSlider.setBlockIncrement(1D);
		
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(logTF);
		HBox hbox2 = new HBox();
		hbox2.getChildren().addAll(numOrientsSlider, numScalesSlider);
		
		
		numOrientsSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setN_t(new_val.doubleValue());
				retina.clear();
				logTF.setText("num orientations = " + new_val.doubleValue());
				try {
//					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
//					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		numScalesSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setN_s(new_val.doubleValue());
				retina.clear();
				logTF.setText("num scales = " + new_val.doubleValue());
				try {
//					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
//					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		
		pane.getChildren().addAll(hbox, hbox2);
		
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



}
