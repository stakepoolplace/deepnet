package RN;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.INode;
import RN.nodes.Node;
import javafx.event.EventHandler;
import javafx.scene.AmbientLight;
import javafx.scene.PerspectiveCamera;
import javafx.scene.PointLight;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.scene.shape.DrawMode;
import javafx.scene.shape.Line;
import javafx.scene.shape.Sphere;

/**
 * @author Eric Marchand
 * 
 */
public class Graphics3D {

	private static Map<Link, Line> shapeLinks = new HashMap<Link, Line>();
	private static Map<INode, Sphere> shapeNodes = new HashMap<INode, Sphere>();
	
	
	private static PointLight pointLight = new PointLight();
	private static AmbientLight ambient = new AmbientLight();
	private static PhongMaterial material = new PhongMaterial();


	final static Graphics3DXForm world = new Graphics3DXForm();
	final static PerspectiveCamera camera = new PerspectiveCamera(true);
	final static Graphics3DXForm cameraXform = new Graphics3DXForm();
	final static Graphics3DXForm cameraXform2 = new Graphics3DXForm();
	final static Graphics3DXForm cameraXform3 = new Graphics3DXForm();

	private static final double AXIS_LENGTH = 250.0;

	final static Graphics3DXForm axisGroup = new Graphics3DXForm();

	private static final double CAMERA_INITIAL_DISTANCE = -350;
	private static final double CAMERA_INITIAL_X_ANGLE = 0.0;
	private static final double CAMERA_INITIAL_Y_ANGLE = 0.0;
	private static final double CAMERA_NEAR_CLIP = 0.1;
	private static final double CAMERA_FAR_CLIP = 10000.0;

	private static final double CONTROL_MULTIPLIER = 0.1;
	private static final double SHIFT_MULTIPLIER = 10.0;
	private static final double MOUSE_SPEED = 2.5;
	private static final double ROTATION_SPEED = 2.0;
	private static final double TRACK_SPEED = 0.3;
	public static final boolean graphics3DActive = false;

	static double mousePosX;
	static double mousePosY;
	static double mouseOldX;
	static double mouseOldY;
	static double mouseDeltaX;
	static double mouseDeltaY;

	public static void initCameraLightAndAxes() {

		buildCamera();
		buildAxes();

		pointLight.setTranslateX(800);
		pointLight.setTranslateY(100);
		pointLight.setTranslateZ(-1100);
		pointLight.setColor(Color.rgb(255, 255, 255, 0.5D));
		pointLight.setLightOn(false);
		
		ambient.setColor(Color.rgb(0, 255, 0, 1));
		ambient.setLightOn(false);

		
		Image cells = new Image("file:/Users/ericmarchand/Documents/workspace_neural/cells.png", true);
		material.setSelfIlluminationMap(cells);
		material.setDiffuseColor(Color.WHITE);
		//material.setBumpMap(new Image("http://www.inserm.fr/var/inserm/storage/images/mediatheque/infr-grand-public/images/images-de-sciences/inserm_52314neuroneaxone4/325924-1-fre-FR/inserm_52314neuroneaxone4.jpg"));
		//material.setDiffuseMap(new Image("http://www.inserm.fr/var/inserm/storage/images/mediatheque/infr-grand-public/images/images-de-sciences/inserm_52314neuroneaxone4/325924-1-fre-FR/inserm_52314neuroneaxone4.jpg"));
		material.setSpecularMap(cells);
		
		world.getChildren().addAll(pointLight, ambient);

	}

	public static void createLayer(ILayer layer) {

		// Shape3D[] meshView = new Shape3D[] { new Box(60, 300, 100) };
		//
		// for (int i = 0; i < meshView.length; ++i) {
		// meshView[i].setMaterial(material);
		// meshView[i].setDrawMode(DrawMode.LINE);
		// meshView[i].setCullFace(CullFace.NONE);
		// meshView[i].setTranslateX(getLayerXYZ(layer)[0]);
		// meshView[i].setTranslateY(getLayerXYZ(layer)[1]);
		// meshView[i].setTranslateZ(getLayerXYZ(layer)[2]);
		// }
		//
		// world.getChildren().addAll(meshView);

	}

	public static Double[] getLayerXYZ(ILayer layer) {
		if(layer == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		if(layer != null && layer.getLayerId() == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		return new Double[] { (layer.getLayerId() + 1D) * -80D - ((layer.getNetwork().getLayers().size() + 1) * -80D) / 2D,
				0D, 0D };
	}

	public static void createArea(Area area) {

		// Shape3D[] meshView = new Shape3D[] { new Box(50, 50, 50) };
		//
		// for (int i = 0; i < meshView.length; ++i) {
		// meshView[i].setMaterial(material);
		// meshView[i].setDrawMode(DrawMode.LINE);
		// meshView[i].setCullFace(CullFace.NONE);
		// meshView[i].setTranslateX(getAreaXYZ(area)[0]);
		// meshView[i].setTranslateY(getAreaXYZ(area)[1]);
		// meshView[i].setTranslateZ(getAreaXYZ(area)[2]);
		// }
		//
		// graph.getChildren().addAll(meshView);

	}

	public static Double[] getAreaXYZ(IArea area) {
		
		if(area == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		if(area != null && area.getAreaId() == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		return new Double[] { getLayerXYZ(area.getLayer())[0],
				getLayerXYZ(area.getLayer())[1] + area.getAreaId() * 60D + 10D, getLayerXYZ(area.getLayer())[2] };
	}

	public static void createNode(Node node) {

		Sphere sphere =  new Sphere(10D);

		sphere.setMaterial(material);
		sphere.setTranslateX(getNodeXYZ(node)[0]);
		sphere.setTranslateY(getNodeXYZ(node)[1]);
		sphere.setTranslateZ(getNodeXYZ(node)[2]);
		sphere.setDrawMode(DrawMode.FILL);
		//sphere.setCullFace(CullFace.BACK);
		
		shapeNodes.put(node, sphere);
		
		world.getChildren().addAll(sphere);

	}

	public static Double[] getNodeXYZ(INode node) {
		
		if(node == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		if(node != null && node.getNodeId() == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		return new Double[] { 
				getAreaXYZ(node.getArea())[0], 
				getAreaXYZ(node.getArea())[1] + node.getNodeId() * 30D,
				getAreaXYZ(node.getArea())[2] };
	}

	public static void createLink(Link link) {

		Line line = shapeLinks.get(link);
		
		if(line != null){
			
			if(link.getTargetNode() == null && link.getSourceNode() != null){
				line.setStartX(getNodeXYZ(link.getSourceNode())[0]);
				line.setStartY(getNodeXYZ(link.getSourceNode())[1]);
			}
			
			if(link.getTargetNode() != null && link.getSourceNode() == null){
				line.setEndX(getNodeXYZ(link.getTargetNode())[0]);
				line.setEndY(getNodeXYZ(link.getTargetNode())[1]);
			}
			
		} else {
			
			if(link.getType() == ELinkType.SELF_NODE_LINK)
				return;
			
			
			if(link.isBias()){
				line = new Line(getNodeXYZ(link.getTargetNode())[0], getNodeXYZ(link.getTargetNode())[1] + 40D,
						getNodeXYZ(link.getTargetNode())[0], getNodeXYZ(link.getTargetNode())[1]);
				shapeLinks.put(link, line);
			
			}
			
			if(link.getSourceNode() != null && link.getTargetNode() == null){
				line = new Line(getNodeXYZ(link.getSourceNode())[0], getNodeXYZ(link.getSourceNode())[1], getNodeXYZ(link.getSourceNode())[0] - 60D, getNodeXYZ(link.getSourceNode())[1]);
				shapeLinks.put(link, line);
			}
			
			if (link.getSourceNode() == null && link.getTargetNode() != null) {
				line = new Line(getNodeXYZ(link.getTargetNode())[0] + 60D, getNodeXYZ(link.getTargetNode())[1], getNodeXYZ(link.getTargetNode())[0], getNodeXYZ(link.getTargetNode())[1]);
				shapeLinks.put(link, line);
			}
			
			line.setSmooth(true);
			
			world.getChildren().addAll(line);
		}
		
		
//		Graphics3DXForm lineWeight = new Graphics3DXForm();
//		
//		
//		Text textnode = new Text("VALUE");
//		textnode.setFont(Font.font("Verdana", 8));
		//textnode.setFill(Paint.valueOf("BLACK"));
		//t.setRotate(180);
		//textnode.setSmooth(true);
//		textnode.setLayoutX( Math.abs(line.getEndX() - line.getStartX()) / 2D - textnode.getLayoutBounds().getMinX());
//		textnode.setLayoutY( Math.abs(line.getEndY() - line.getStartY()) / 2D - textnode.getLayoutBounds().getMinY());
		
//		lineWeight.getChildren().addAll(line, textnode);
		
		

	}

	public static Double[] getLinkXYZ(Link link) {
		
		if(link == null)
			return new Double[] { 
					0D, 
					0D,
					0D };
		
		return new Double[] { getNodeXYZ(link.getSourceNode())[0], getNodeXYZ(link.getSourceNode())[1],
				getNodeXYZ(link.getSourceNode())[2] };
	}

	public static void buildCamera() {
		world.getChildren().add(cameraXform);
		cameraXform.getChildren().add(cameraXform2);
		cameraXform2.getChildren().add(cameraXform3);
		cameraXform3.getChildren().add(camera);
		cameraXform3.setRotateZ(180.0);
		cameraXform2.setRotateY(180.0);

		camera.setNearClip(CAMERA_NEAR_CLIP);
		camera.setFarClip(CAMERA_FAR_CLIP);
		camera.setTranslateZ(CAMERA_INITIAL_DISTANCE);
		cameraXform.ry.setAngle(CAMERA_INITIAL_Y_ANGLE);
		cameraXform.rx.setAngle(CAMERA_INITIAL_X_ANGLE);
	}

	public static void handleMouse(Scene scene) {

		scene.setOnMousePressed(new EventHandler<MouseEvent>() {
			@Override
			public void handle(MouseEvent me) {
				mousePosX = me.getSceneX();
				mousePosY = me.getSceneY();
				mouseOldX = me.getSceneX();
				mouseOldY = me.getSceneY();
			}
		});
		scene.setOnMouseDragged(new EventHandler<MouseEvent>() {
			@Override
			public void handle(MouseEvent me) {
				mouseOldX = mousePosX;
				mouseOldY = mousePosY;
				mousePosX = me.getSceneX();
				mousePosY = me.getSceneY();
				mouseDeltaX = (mousePosX - mouseOldX);
				mouseDeltaY = (mousePosY - mouseOldY);

				double modifier = 1.0;
				double modifierFactor = 0.1;

				if (me.isControlDown()) {
					modifier = CONTROL_MULTIPLIER;
				}
				if (me.isShiftDown()) {
					modifier = SHIFT_MULTIPLIER;
				}
				if (me.isPrimaryButtonDown()) {
					cameraXform.ry.setAngle(
							cameraXform.ry.getAngle() - mouseDeltaX * modifierFactor * modifier * ROTATION_SPEED); //
					cameraXform.rx.setAngle(
							cameraXform.rx.getAngle() + mouseDeltaY * modifierFactor * modifier * ROTATION_SPEED); // -
				} else if (me.isSecondaryButtonDown()) {
					double z = camera.getTranslateZ();
					double newZ = z + mouseDeltaX * MOUSE_SPEED * modifier;
					camera.setTranslateZ(newZ);
				} else if (me.isMiddleButtonDown()) {
					cameraXform2.t.setX(cameraXform2.t.getX() + mouseDeltaX * MOUSE_SPEED * modifier * TRACK_SPEED); // -
					cameraXform2.t.setY(cameraXform2.t.getY() + mouseDeltaY * MOUSE_SPEED * modifier * TRACK_SPEED); // -
				}
			}
		}); // setOnMouseDragged
	} // handleMouse

	public static void handleKeyboard(Scene scene) {

		scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent event) {
				switch (event.getCode()) {
				case C:
					cameraXform2.t.setX(0.0);
					cameraXform2.t.setY(0.0);
					cameraXform.ry.setAngle(CAMERA_INITIAL_Y_ANGLE);
					cameraXform.rx.setAngle(CAMERA_INITIAL_X_ANGLE);
					break;
				case A:
					axisGroup.setVisible(!axisGroup.isVisible());
					break;
				case L:
					pointLight.setLightOn(!pointLight.isLightOn());
					break;
				case M:
					ambient.setLightOn(!ambient.isLightOn());
					break;
				case SPACE:
					scene.setFill(scene.getFill() == Color.BLACK ? Color.WHITE : Color.BLACK);
					break;
				// case V:
				// moleculeGroup.setVisible(!moleculeGroup.isVisible());
				// break;
				default:
					break;
				}
			}
		});
	}

	public static void buildAxes() {

		final PhongMaterial redMaterial = new PhongMaterial();
		redMaterial.setDiffuseColor(Color.DARKRED);
		redMaterial.setSpecularColor(Color.RED);

		final PhongMaterial greenMaterial = new PhongMaterial();
		greenMaterial.setDiffuseColor(Color.DARKGREEN);
		greenMaterial.setSpecularColor(Color.GREEN);

		final PhongMaterial blueMaterial = new PhongMaterial();
		blueMaterial.setDiffuseColor(Color.DARKBLUE);
		blueMaterial.setSpecularColor(Color.BLUE);

		final Box xAxis = new Box(AXIS_LENGTH, 1, 1);
		final Box yAxis = new Box(1, AXIS_LENGTH, 1);
		final Box zAxis = new Box(1, 1, AXIS_LENGTH);

		xAxis.setMaterial(redMaterial);
		yAxis.setMaterial(greenMaterial);
		zAxis.setMaterial(blueMaterial);

		axisGroup.getChildren().addAll(xAxis, yAxis, zAxis);
		axisGroup.setVisible(false);
		
		
		world.getChildren().addAll(axisGroup);
	}

	public static void clearShapes() {
		
		shapeLinks.clear();
		shapeNodes.clear();
		world.getChildren().removeIf(new Predicate<javafx.scene.Node>() {

			@Override
			public boolean test(javafx.scene.Node node) {
				return !(node instanceof javafx.scene.PointLight) && !(node instanceof javafx.scene.AmbientLight) && !(node instanceof RN.Graphics3DXForm);
			}

		});
	}
	
	public static void setWeightOnLink(Link link){
		
		Line line = shapeLinks.get(link);
		if(line == null)
			return;
		
		line.setFill(Color.TRANSPARENT);
		
		if(link.getWeight() < 0D){
			line.setStroke(Color.color(1D, 0D, 0D));
		}else if(link.getWeight() > 0D){
			line.setStroke(Color.color(0D, 1D, 0D));
		}else if(link.getWeight() == 0D)
			line.setStroke(Color.WHITE);
		
		line.setStrokeWidth(Math.abs(link.getWeight() / 4D));
		
	}
	
	public static void setOutputValueOnNode(INode node){
		Sphere sphere = shapeNodes.get(node);
		
		if(sphere != null && Math.abs(node.getComputedOutput()) <= 1D ){
			
			material.setDiffuseColor(Color.rgb((node.getComputedOutput() < 0D ? (int) (-node.getComputedOutput() * 255) : 0), 0, (node.getComputedOutput() > 0D ? (int) (node.getComputedOutput() * 255) : 0)));

			sphere.setMaterial(material);
			sphere.setRadius(10D * Math.abs(node.getComputedOutput()));
		}
	}

}
