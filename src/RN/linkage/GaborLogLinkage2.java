package RN.linkage;

import RN.IAreaSquare;
import RN.ILayer;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.FilterLinkage.FilterIndex;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.utils.StatUtils;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * @author Eric Marchand
 *
 */
public class GaborLogLinkage2 extends FilterLinkage {
	
	private Slider maskSizeSlider = null;
	private Slider numOrientsSlider = null;
	private Slider numScalesSlider = null;
	
	private Slider idThetaSlider = null;
	private Slider idScaleSlider = null;

	//Number of orientations (between 3 to 20) 8 is a typical value
	private Double numOrients = null;
	private Double t = null;
	
	//Number of scales of the multiresolution scheme
	private Double numScales = null;
	private Double s = null;
	
	private Integer maskSize = null;
	private Integer maskSizeTotal = null;
	
	private GaborFilter gaborFilter = null;
	private float[] data = null;
	
	private int widthPx;
	private int heightPx;
	
	
	private Double[][] gaborReal;
	private Double[][] gaborImag;

	public GaborLogLinkage2() {
	}
	

	
	public void initParameters() {
		
		if(params.length != 4)
			throw new RuntimeException("Missing Gabor parameters'");
		
		
		numOrients = params[0];
		numScales = params[1];
		t = params[2];
		s = params[3];
		
		maskSize = 2;
		
		computeMaskSize(maskSize);
		
//		widthPx = ((IAreaSquare) thisArea).getWidthPx();
//		heightPx = ((IAreaSquare) thisArea).getHeightPx();
//		
//		if(data == null){
//			gaborFilter = new GaborFilter(16, new double[] {t * Math.PI /numOrients}, 0, 0.5, 1, widthPx, heightPx);
//			data = gaborFilter.getKernel().getKernelData(null);
//		}
		
		
	}
	
	@Override
	public double getUnLinkedSigmaPotentials(INode thisNode){
		
		// somme des entrees pondérées
		SigmaWi sigmaWI = new SigmaWi();
		
		IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		//initFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode) thisNode, widthPx, heightPx, data, getNumOrients(), getNumScales(), getT());
		
		initFilter(this, ID_FILTER_GABOR_LOG, ESamples.LOG_GABOR, 0.0001D, (IPixelNode) thisNode, subArea, getNumOrients(), getNumScales(), getT(), getS());
		
		subArea.applyConvolutionFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode) thisNode, sigmaWI);
			
		
		
		return sigmaWI.value();
	}
	
	@Override
	public void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) {
		
		
			IAreaSquare subArea = (IAreaSquare) getLinkedArea();
			
			//initFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode) thisNode, widthPx, heightPx, data, getNumOrients(), getNumScales(), getT());
			
			initFilter(this, ID_FILTER_GABOR_LOG, ESamples.LOG_GABOR, 0.0001D, (IPixelNode) thisNode, subArea, getNumOrients(), getNumScales(), getT(), getS());
			
			subArea.applyConvolutionFilter(this, ID_FILTER_GABOR_LOG, (IPixelNode)thisNode, 0.001f);
		
	}
	
	
	private void computeMaskSize(int maskSize){
		this.maskSize = maskSize;
		this.maskSizeTotal = 2 * maskSize + 1; // gaborReal and gaborImag = new Double[maskSizeTotal][maskSizeTotal]
	}
	
	@Override
	public Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params) {
		
		//IAreaSquare subArea = (IAreaSquare) getLinkedArea();
		
		if(gaborReal == null || gaborImag == null){
			
			gaborReal = new Double[maskSizeTotal][maskSizeTotal];
			gaborImag = new Double[maskSizeTotal][maskSizeTotal];
			
			gaborFilter(gaborReal, gaborImag, numScales.intValue(), numOrients.intValue(), s.intValue(), t.intValue(), true);
			
            double real_weight = StatUtils.absoluteSum(gaborReal);
            double imag_weight = StatUtils.absoluteSum(gaborImag);
            
            real_weight /= maskSizeTotal * maskSizeTotal;
            imag_weight /= maskSizeTotal * maskSizeTotal;
            
            for(int i = 0; i < maskSizeTotal; i++) {
                for(int j = 0; j < maskSizeTotal; j++) {
                	gaborReal[i][j] -= real_weight;
                	gaborImag[i][j] -= imag_weight;
                }
             }
            
		}
		
		if(sublayerNode.getX() >= maskSizeTotal || sublayerNode.getY() >= maskSizeTotal)
			return 0D;
		
		
		return Math.sqrt(Math.pow(gaborReal[ sublayerNode.getX() ][ sublayerNode.getY() ], 2) + Math.pow(gaborImag[ sublayerNode.getX() ][ sublayerNode.getY() ], 2));
		
		//return Math.sqrt(Math.pow( gaborReal[ sublayerNode.getX() ][ sublayerNode.getY() ]/real_weight, 2.0 ) + Math.pow( gaborImag[ sublayerNode.getX() ][ sublayerNode.getY() ]/imag_weight, 2.0 ) );
		
        //compute the magnitude of the complex
        //filtering results
//        int data_index = 0;
//        for(int r = 0; r < image.height; r++ ) {
//                for(int c = 0; c < image.width; c++, data_index++ ) {
//                        features.data[ data_index ][ filter_index ] = Math.sqrt(
//                                Math.pow( real[ r ][ c ]/real_weight, 2.0 ) +
//                                Math.pow( imag[ r ][ c ]/imag_weight, 2.0 ) );
//                }
//        }
		
		// Calcul du filtre Log-Gabor
//		return  InputSample.getInstance().compute(
//				filterFunction, 
//				(double) subArea.getWidthPx(),
//				(double) sublayerNode.getX(), // - subArea.getWidthPx() / 2D) * 4D,
//				(double) sublayerNode.getY(), // - subArea.getHeightPx() / 2D) * 4D,
//				(double) subArea.getNodeCenterX(),
//				(double) subArea.getNodeCenterY(),
//				getNumScales(),
//				getS(),
//				getNumOrients(),
//				getT()
//				);
		
		
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
	
	  //method to generate Gabor filter masks in time domain
	  public static void gaborFilter(Double[][] gaborReal, Double[][] gaborImag,  int numScales, int numOrients, int s, int n, boolean removeDC) {

		  	double freqLower = 0.15;	//lower frequency
		  	double freqUpper = 0.4;  //upper frequency
		  
	          double freqBase, a, u0, z, v, x0, y0, g, t1, t2, m;
	          int x, y, side;

	          freqBase = freqUpper / freqLower;
	          a = Math.pow( freqBase, 1.0 / (double) ( numScales - 1 ) );

	          u0 = freqUpper / Math.pow( a, (double) ( numScales - s ) );

	          v = Math.pow( 0.6 / freqUpper * Math.pow( a, (double) ( numScales - s ) ), 2.0 );

	          t1 = Math.cos( (double) Math.PI / numOrients * ( n - 1.0 ) );
	          t2 = Math.sin( (double) Math.PI / numOrients * ( n - 1.0 ) );

	          side = (int) ( gaborReal[0].length - 1 ) / 2;

	          for( x = 0; x < ( 2 * side + 1 ); x++ ) {
	                  for( y = 0; y < ( 2 * side + 1 ); y++ ) {
	                          x0 = (double) ( x - side) * t1 + (double) ( y - side ) * t2;
	                          y0 = (double) - ( x - side ) * t2 + (double) ( y - side ) * t1;
	                          g = 1.0 / ( 2.0 * Math.PI * v ) * Math.pow( a, (double) ( numScales - s ) ) * Math.exp( - 0.5 * ( x0 * x0 + y0 * y0 ) / v );
	                          gaborReal[x][y] = g * Math.cos( 2.0 * Math.PI * u0 * x0 );
	                          gaborImag[x][y] = g * Math.sin( 2.0 * Math.PI * u0 * x0 );
	                  }
	          }

	          //if removeDC flag is set, then remove the average value
	          //from the real part of Gabor
	          if( removeDC  ) {

	                  m = 0;
	                  for( x = 0; x < ( 2 * side + 1 ); x++ )
	                          for( y = 0; y < ( 2 * side + 1 ); y++ )
	                                  m += Math.abs(gaborReal[x][y]);

	                  m /= Math.pow( (double) 2.0 * side + 1, 2.0 );

	                  for( x = 0; x < ( 2 * side + 1 ); x++ )
	                          for( y = 0; y < ( 2 * side + 1 ); y++ )
	                                  gaborReal[x][y] -= m;

	          }

	  }
	
	public void addGraphicInterface(Pane pane) {
		
		TextField maskSizeTF = new TextField("mask size total = " + getMaskSizeTotal());
		
		maskSizeSlider = new Slider(0D, 8D, maskSize);
		maskSizeSlider.setBlockIncrement(0.05D);
		
		numOrientsSlider = new Slider(0D, 60D, numOrients);
		numOrientsSlider.setBlockIncrement(1D);
		
		numScalesSlider = new Slider(0D, 8D, numScales);
		numScalesSlider.setBlockIncrement(1D);
		
		idThetaSlider = new Slider(0D, numOrients, t);
		idThetaSlider.setBlockIncrement(1D);
		
		idScaleSlider = new Slider(0D, numScales, s);
		idScaleSlider.setBlockIncrement(1D);
		
		
		
		HBox hbox = new HBox();
		hbox.getChildren().addAll(maskSizeSlider, maskSizeTF);
		HBox hbox2 = new HBox();
		hbox2.getChildren().addAll(numOrientsSlider, numScalesSlider);
		HBox hbox3 = new HBox();
		hbox3.getChildren().addAll(idThetaSlider, idScaleSlider);
		
		
		maskSizeSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				computeMaskSize(new_val.intValue());
				maskSizeTF.setText("mask size = " + new_val.doubleValue());
				try {
					gaborReal = null;
					gaborImag = null;
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		numOrientsSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setNumOrients(new_val.doubleValue());
				idThetaSlider.setMax(new_val.intValue());
				maskSizeTF.setText("num orientations = " + new_val.doubleValue());
				try {
					gaborReal = null;
					gaborImag = null;
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
					FilterLinkage.removeFilter(idx);
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
				setNumScales(new_val.doubleValue());
				idScaleSlider.setMax(new_val.intValue());
				maskSizeTF.setText("num scales = " + new_val.doubleValue());
				try {
					gaborReal = null;
					gaborImag = null;
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		idThetaSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setT(new_val.doubleValue());
				maskSizeTF.setText("id theta = " + new_val.doubleValue());
				try {
					gaborReal = null;
					gaborImag = null;
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		idScaleSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				setS(new_val.doubleValue());
				maskSizeTF.setText("id scale = " + new_val.doubleValue());
				try {
					gaborReal = null;
					gaborImag = null;
					FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_GABOR_LOG);
					FilterLinkage.removeFilter(idx);
					Double[] outputs = null;
					thisArea.propagation(false, outputs);
					((IAreaSquare) thisArea).showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		Button showFilterBtn = new Button("show filter");
		
		showFilterBtn.setTooltip(new Tooltip("show filter"));
		
		
		HBox hbox4 = new HBox();
		hbox4.getChildren().addAll(showFilterBtn);
		
		showFilterBtn.setOnAction(new EventHandler<ActionEvent>(){

			@Override
			public void handle(ActionEvent event) {
				try {
					//FilterIndex idx = new FilterIndex(thisArea.getIdentification(), ID_FILTER_CONVOLUTION);
					((IAreaSquare)thisArea).getFilter(FilterLinkage.ID_FILTER_GABOR_LOG).filterToImage(8);
					//thisArea.showImageArea();
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			
		});
		
		
		pane.getChildren().addAll(hbox, hbox2, hbox3, hbox4);
		
		
		
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



	public Double getNumOrients() {
		return numOrients;
	}



	public void setNumOrients(Double numOrients) {
		this.numOrients = numOrients;
	}



	public Double getNumScales() {
		return numScales;
	}



	public void setNumScales(Double numScales) {
		this.numScales = numScales;
	}



	public Integer getMaskSizeTotal() {
		return maskSizeTotal;
	}



	public void setMaskSizeTotal(Integer maskSizeTotal) {
		this.maskSizeTotal = maskSizeTotal;
	}



	public Integer getMaskSize() {
		return maskSize;
	}



	public void setMaskSize(Integer maskSize) {
		this.maskSize = maskSize;
	}

}
