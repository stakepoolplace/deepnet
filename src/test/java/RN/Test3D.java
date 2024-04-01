package RN;

	
	import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.PointLight;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.scene.shape.CullFace;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.DrawMode;
import javafx.scene.shape.Shape3D;
import javafx.scene.shape.Sphere;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Translate;
import javafx.stage.Stage;
	 
	public class Test3D extends Application {
		
//	    @Override 
//	    public void start(Stage stage) {
//	    	
//	        Group root = test();
//	         
//	        Scene scene = new Scene(root, 800, 400, true);
//	        scene.setFill(Color.rgb(10, 10, 40));
//	        scene.setCamera(new PerspectiveCamera(false));
//	        stage.setScene(scene);
//	        stage.show();
//	    }
	    
	    public void start(Stage primaryStage) {
	        Sphere sphere = new Sphere(100);
	        sphere.setMaterial(new PhongMaterial(Color.BLUE));
	        Box box = new Box(50,50,50);
	        box.setMaterial(new PhongMaterial(Color.RED));
	        box.setTranslateX(300);
	        Cylinder cylinder = new Cylinder(2, 300);
	        cylinder.setMaterial(new PhongMaterial(Color.GREEN));
	        // Transformations applied:
	        cylinder.getTransforms().addAll(new Translate(150, 0, 0), new Rotate(90, Rotate.Z_AXIS));

	        Group group = new Group(cylinder, sphere, box);
	        StackPane root = new StackPane(group);

	        Scene scene = new Scene(root, 600, 400);

	        primaryStage.setScene(scene);
	        primaryStage.show();

	        // export as single mesh
//	        EquivalentMesh equivalentMesh = new EquivalentMesh(root);
//	        equivalentMesh.export("group");
	    }

		private Group test() {
			PhongMaterial material = new PhongMaterial();
	        material.setDiffuseColor(Color.LIGHTGRAY);
	        material.setSpecularColor(Color.rgb(30, 30, 30));
	 
	        Shape3D[] meshView = new Shape3D[] {
	            new Box(200, 200, 200),
	            new Sphere(100),
	            new Cylinder(100, 200),
	        };
	 
	        for (int i=0; i!=3; ++i) {
	            meshView[i].setMaterial(material);
	            meshView[i].setTranslateX((i) * 220 + 180);
	            meshView[i].setTranslateY(200);
	            meshView[i].setTranslateZ(20);
	            meshView[i].setDrawMode(DrawMode.FILL);
	            meshView[i].setCullFace(CullFace.BACK);
	        };
	 
	        PointLight pointLight = new PointLight(Color.ANTIQUEWHITE);
	        pointLight.setTranslateX(800);
	        pointLight.setTranslateY(-100);
	        pointLight.setTranslateZ(-1000);
	 
	        Button button = new Button("Hello");
	        Group root = new Group(meshView);
	        root.getChildren().addAll(pointLight, button);
			return root;
		}
	 
	    public static void main(String[] args) {
	        launch(args);
	    }
	}

