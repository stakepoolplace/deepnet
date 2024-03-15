package RN.nodes;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

import RN.algoactivations.EActivation;
import RN.linkage.Filter;
import RN.links.Link;
import RN.links.Weight;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.geometry.Insets;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Tooltip;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class ImageNode extends PixelNode {
	
	public Stage stage = new Stage();
	public Scene scene2 = null;
	public Canvas canvas;
	public GraphicsContext gc;
	public PixelWriter pixelWriter;
	public PixelFormat<ByteBuffer> pixelFormat = PixelFormat.getByteRgbInstance();

	// Image Data
	private int IMAGE_WIDTH = 100;
	private int IMAGE_HEIGHT = 100;
	public byte imageData[] = null;
	
	private Boolean red = Boolean.TRUE;
	private Boolean green = Boolean.TRUE;
	private Boolean blue = Boolean.TRUE;
	
	private CheckBox redCB = null;
	private CheckBox greenCB = null;
	private CheckBox blueCB = null;
	private CheckBox negativeValuesCB = null;
	
	private Boolean negativeValuesActivated = Boolean.FALSE;
	
	private INode node = null;
	
	//private Area area = null;
	
	
	public ImageNode() {
		super();
		this.nodeType = ENodeType.IMAGE;
		imageData = new byte[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
	}

	
	public ImageNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.IMAGE;
		imageData = new byte[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
	}
	
	public ImageNode(EActivation activationFx, Integer widthPx, Integer heightPx) {
		super(activationFx);
		this.nodeType = ENodeType.IMAGE;
		IMAGE_WIDTH = widthPx;
		IMAGE_HEIGHT = heightPx;
		imageData = new byte[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
	}

	public void initImageData(){
		imageData = new byte[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
	}
	
	public void scaleImage(double multiplier) {

		if (gc == null) {
			initImageScene();
		}

		gc.getCanvas().setScaleX(multiplier);
		gc.getCanvas().setScaleY(multiplier);

		gc.getCanvas().setTranslateX(scene2.getWidth() / 2 - IMAGE_WIDTH / 2);
		gc.getCanvas().setTranslateY(scene2.getHeight() / 2 - IMAGE_HEIGHT / 2);
	}
	
	public void initImageData(int width, int height){
		imageData = new byte[ width * height * 3];
		IMAGE_WIDTH = width;
		IMAGE_HEIGHT = height;
	}

	public void initImageScene(INode node) {
		canvas = new Canvas(IMAGE_WIDTH, IMAGE_HEIGHT);
		gc = canvas.getGraphicsContext2D();
		pixelWriter = gc.getPixelWriter();
		try {
			showDataStage(node);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void initImageScene() {
		initImageScene(null);
	}
	
	public void showImage(PixelNode node) {

		this.node = node;
		
		insertData(node);

		drawImageData(node);
	}
	
	@Override
	public void showImage(INode node) {
		
		this.node = node;
		
		insertData(node);

		drawImageData(node);
	}
	
	
	public void insertData(INode node){
		
		
		Integer index = null;
		double filterIntensity = 0D;
		double imagePixelsIntensity = 0D;
		
		double filteredPixelLuminance = 0D;
		
		for (Link link : node.getInputs()) {

			
			filterIntensity = link.getWeight();
			imagePixelsIntensity = link.getValue();

			filteredPixelLuminance = filterIntensity * imagePixelsIntensity;
			
			filteredPixelLuminance *= 255D;
			filterIntensity *= 255D;
			imagePixelsIntensity *= 255D;
			
			index = (link.getSourceNode() != null ? link.getSourceNode().getNodeId() : node.getNodeId());

			if(negativeValuesActivated){
				
				if(red)
					imageData[index * 3 ] += Byte.valueOf((byte) imagePixelsIntensity);
				
				if(green)
					imageData[index * 3 + 1] +=  (byte) (filteredPixelLuminance < 0D ? -filteredPixelLuminance > 255D ? 255D : -filteredPixelLuminance : 0D);
				
				if(blue)
					imageData[index * 3 + 2] += (byte)  (filteredPixelLuminance <= 0D ? 0D : filteredPixelLuminance > 255D ? 255D : filteredPixelLuminance);
				

			}else{
			
				// rouge + vert = jaune
				// bleu + rouge = magenta
				// vert + bleu = cyan
					
				filterIntensity = Math.min(255D, Math.max(0D, filterIntensity));
				imagePixelsIntensity = Math.min(255D, Math.max(0D, imagePixelsIntensity));
				filteredPixelLuminance = Math.min(255D, Math.max(0D, filteredPixelLuminance));
				
				if(red)
					imageData[index * 3] += (byte)  imagePixelsIntensity;
				
				if(green)
					imageData[index * 3 + 1] += (byte) filteredPixelLuminance;
				
				if(blue)
					imageData[index * 3 + 2] += (byte) filterIntensity;
				
				}
			
		}
		
	}
	
	public void insertDataArea(){
		
		
		double imagePixelsIntensity = 0D;
		
		List<INode> nodeList = area.getNodes();
		INode node = null;
		for (int index = 0; index < nodeList.size(); index++) {
			
			node = area.getNode(index);
			
			imagePixelsIntensity = node.getComputedOutput();
			
			imagePixelsIntensity *= 255D;
			
			

			if(negativeValuesActivated){
				
				if(red)
					imageData[index * 3 ] += (byte) imagePixelsIntensity;
				
				if(green)
					imageData[index * 3 + 1] +=  (byte) (imagePixelsIntensity < 0D ? -imagePixelsIntensity > 255D ? 255D : -imagePixelsIntensity : 0D);
				
				if(blue)
					imageData[index * 3 + 2] += (byte)  (imagePixelsIntensity <= 0D ? 0D : imagePixelsIntensity > 255D ? 255D : imagePixelsIntensity);
				

			}else{
			
				// rouge + vert = jaune
				// bleu + rouge = magenta
				// vert + bleu = cyan
					
				imagePixelsIntensity = Math.min(255D, Math.max(0D, imagePixelsIntensity));
				
				if(red)
					imageData[index * 3] += (byte) 0D;
				
				if(green)
					imageData[index * 3 + 1] += (byte) imagePixelsIntensity;
				
				if(blue)
					imageData[index * 3 + 2] += (byte) 0D;
			
			}
			
		}
		
	}
	
	public void insertDataFilter(Filter filter) {

		double imagePixelsIntensity = 0D;

		int index = 0;
		Double value = null;

		for (int idy = 0; idy < filter.getHeight(); idy++) {
			for (int idx = 0; idx < filter.getWidth(); idx++) {

				value = filter.getValue(idx, idy);

				imagePixelsIntensity = value;

				imagePixelsIntensity *= 255D;

				if (negativeValuesActivated) {

					if (red)
						imageData[index * 3] += (byte) imagePixelsIntensity;

					if (green)
						imageData[index * 3 + 1] += (byte) (imagePixelsIntensity < 0D ? -imagePixelsIntensity > 255D ? 255D : -imagePixelsIntensity : 0D);

					if (blue)
						imageData[index * 3 + 2] += (byte) (imagePixelsIntensity <= 0D ? 0D : imagePixelsIntensity > 255D ? 255D : imagePixelsIntensity);

				} else {

					// rouge + vert = jaune
					// bleu + rouge = magenta
					// vert + bleu = cyan
//		              data[i*3]= (byte)(imageData[i]>>16&0xff);
//		              data[i*3+1]= (byte)(imageData[i]>>8&0xff);
//		              data[i*3+2]= (byte)(imageData[i]&0xff);

					imagePixelsIntensity = Math.min(255D, Math.max(0D, imagePixelsIntensity));

					if (red)
						imageData[index * 3] += (byte) 0D;

					if (green)
						imageData[index * 3 + 1] += (byte) imagePixelsIntensity;

					if (blue)
						imageData[index * 3 + 2] += (byte) 0D;

				}
				
				index++;
			}

		}

	}
	
	public void insertDataArray(Weight[][] weights) {

		double imagePixelsIntensity = 0D;

		int index = 0;
		Double value = null;

		for (int idy = 0; idy < weights.length; idy++) {
			for (int idx = 0; idx < weights[0].length; idx++) {

				try{
					value = weights[idx][idy].getWeight();
				}catch(Throwable t){
					value = 0D;
				}

				imagePixelsIntensity = value;

				imagePixelsIntensity *= 255D;

				if (negativeValuesActivated) {

					if (red)
						imageData[index * 3] += (byte) imagePixelsIntensity;

					if (green)
						imageData[index * 3 + 1] += (byte) (imagePixelsIntensity < 0D ? -imagePixelsIntensity > 255D ? 255D : -imagePixelsIntensity : 0D);

					if (blue)
						imageData[index * 3 + 2] += (byte) (imagePixelsIntensity <= 0D ? 0D : imagePixelsIntensity > 255D ? 255D : imagePixelsIntensity);

				} else {

					// rouge + vert = jaune
					// bleu + rouge = magenta
					// vert + bleu = cyan
//		              data[i*3]= (byte)(imageData[i]>>16&0xff);
//		              data[i*3+1]= (byte)(imageData[i]>>8&0xff);
//		              data[i*3+2]= (byte)(imageData[i]&0xff);

					imagePixelsIntensity = Math.min(255D, Math.max(0D, imagePixelsIntensity));

					if (red)
						imageData[index * 3] += (byte) 0D;

					if (green)
						imageData[index * 3 + 1] += (byte) imagePixelsIntensity;

					if (blue)
						imageData[index * 3 + 2] += (byte) 0D;

				}
				
				index++;
			}

		}

	}
	
	public void drawImageData(INode node) {

		if(gc == null){
			initImageScene(node);
		}
		
		pixelWriter.setPixels(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, pixelFormat, imageData, 0, IMAGE_WIDTH * 3);

		// Add drop shadow effect
		// gc.applyEffect(new DropShadow(20, 20, 20, Color.GRAY));
	}
	
	public void setPixelAt(int x, int y, Color color){
		
		pixelWriter.setColor(x, y, color);
		
	}
	
	private void showDataStage(INode node) throws IOException {

		int sceneH = 500;
		int sceneW = 500;
		
//		stage.setFullScreen(true);
		
		if(node != null)
			stage.setTitle("Neuron :" + ((INode) node).getIdentification() + " - type : " + ((INode) node).getNodeType().name());

		
		Group group = new Group();
		group.setAutoSizeChildren(true);
		scene2 = new Scene(group, sceneH, sceneW, Color.WHITE);
		stage.setScene(scene2);
		group.getChildren().add(canvas);
		
		HBox hboxCanvas = new HBox();
		hboxCanvas.getChildren().add(gc.getCanvas());
		
		VBox vbox = new VBox();
		
		// Common interface
		// Lightning
		
		HBox hbox = new HBox();
		
		Slider intensitySlider = new Slider(-20D, 20D, 0D);
		intensitySlider.setBlockIncrement(0.1);
		intensitySlider.setTooltip(new Tooltip("Light +/-0.1"));
		
		intensitySlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				if(!old_val.equals(new_val)){
					
					byte[] newImageData = new byte[imageData.length];
					System.arraycopy(imageData, 0, newImageData, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3);
					for(int idx = 0; idx < IMAGE_WIDTH * IMAGE_HEIGHT * 3; idx++){
						if(new_val.floatValue() >= 1)	
							newImageData[idx] = (byte)  Math.min(255, newImageData[idx] * new_val.floatValue()) ;
						
						if(new_val.floatValue() < 0)
							newImageData[idx] = (byte)  (newImageData[idx] / new_val.floatValue()) ;
					}
					pixelWriter.setPixels(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, pixelFormat, newImageData, 0, IMAGE_WIDTH * 3);
					System.out.println("Light : x"+ new_val.floatValue());
				}
			}
		});
		
		Slider scaleSlider = new Slider(1D, 10D, 1D);
		scaleSlider.setBlockIncrement(1);
		scaleSlider.setTooltip(new Tooltip("Scale"));
		
		
		scaleSlider.valueProperty().addListener(new ChangeListener<Number>() {
			public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
				if(!old_val.equals(new_val)){
					scaleImage(new_val.intValue());
				}
			}
		});
		
		hbox.getChildren().addAll(intensitySlider, scaleSlider);
		
		redCB = new CheckBox();
		redCB.setSelected(red);
		greenCB = new CheckBox();
		greenCB.setSelected(green);
		blueCB = new CheckBox();
		blueCB.setSelected(blue);
		
		negativeValuesCB = new CheckBox();
		negativeValuesCB.setSelected(negativeValuesActivated);
		negativeValuesCB.setText("red -> values,  green -> negatives value*weight,  blue ->positives value*weight");
		
		HBox negativeValuesHbox = new HBox(negativeValuesCB);
		negativeValuesHbox.setPadding(new Insets(5D));


		redCB.setText("Inputs value (Red)");
		greenCB.setText("Inputs value*weight (Green)");
		blueCB.setText("Inputs weight (Blue)");
		HBox rvbHbox = new HBox(redCB, greenCB, blueCB);
		rvbHbox.setPadding(new Insets(5D));

		redCB.selectedProperty().addListener(new ChangeListener<Boolean>() {

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				red = newValue;
				initImageData();
				insertDataArea();
				drawImageData(null);

			}

		});

		greenCB.selectedProperty().addListener(new ChangeListener<Boolean>() {

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				green = newValue;
				initImageData();
				insertDataArea();
				drawImageData(null);

			}

		});

		blueCB.selectedProperty().addListener(new ChangeListener<Boolean>() {

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				blue = newValue;
				initImageData();
				insertDataArea();
				drawImageData(null);

			}

		});
		
		negativeValuesCB.selectedProperty().addListener(new ChangeListener<Boolean>() {

			@Override
			public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
				negativeValuesActivated = newValue;
				initImageData();
				insertDataArea();
				drawImageData(null);

			}

		});
		
		
		
		vbox.getChildren().addAll(hbox, rvbHbox, negativeValuesHbox);
		
		
		
		if(this.getArea() != null){
			if(!this.getArea().getNodes().isEmpty())
				this.getArea().getNodes().get(0).addGraphicInterface(vbox);
			
			this.area.getLinkage().addGraphicInterface(vbox);
		}
		
		if (node != null) {
			
			node.addGraphicInterface(vbox);

			Slider nodeSlider = new Slider(0D, node.getArea().getNodeCount() - 1, node.getNodeId());
			nodeSlider.setBlockIncrement(1);
			nodeSlider.valueProperty().addListener(new ChangeListener<Number>() {
				public void changed(ObservableValue<? extends Number> ov, Number old_val, Number new_val) {
					try {
						INode newNode = node.getArea().getNode(new_val.intValue());
						stage.setTitle(
								"Neuron :" + newNode.getIdentification() + " - type : " + newNode.getNodeType().name());
						insertData(newNode);
						drawImageData(newNode);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			});

			Button scanAllNodes = new Button("scan all nodes");
			scanAllNodes.pressedProperty().addListener(new ChangeListener<Boolean>() {

				@Override
				public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
					if(oldValue == false && newValue == true){
						List<INode> nodes = node.getArea().getNodes();
						INode currentnode = null;
						for (int idx = 0; idx < nodes.size(); idx++) {
							currentnode = nodes.get(idx);
							stage.setTitle("Neuron :" + currentnode.getIdentification() + " - type : "
									+ currentnode.getNodeType().name());
							insertData(currentnode);
						}

						drawImageData(currentnode);
					}

				}

			});



			vbox.getChildren().addAll(nodeSlider, scanAllNodes);

		}
		
		vbox.getChildren().addAll(hboxCanvas);
		vbox.setPadding(new Insets(5D));
		
		group.getChildren().addAll(vbox);
        
        stage.sizeToScene();
        
        stage.show();
        
	}

	public byte[] getImageData() {
		return imageData;
	}
	
	public byte getImageData(int idx) {
		return imageData[idx];
	}

	public void setImageData(byte[] imageData) {
		this.imageData = imageData;
	}
	
	public void setImageData(int idx, byte imageData) {
		this.imageData[idx] = imageData;
	}

	public Boolean getNegativeValuesActivated() {
		return negativeValuesActivated;
	}

	public void setNegativeValuesActivated(Boolean negativeValuesActivated) {
		this.negativeValuesActivated = negativeValuesActivated;
	}

	public INode getNode() {
		return node;
	}

	public void setNode(PixelNode node) {
		this.node = node;
	}

	public Stage getStage() {
		return stage;
	}

	public void setStage(Stage stage) {
		this.stage = stage;
	}



}
