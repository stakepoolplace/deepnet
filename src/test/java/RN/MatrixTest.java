package RN;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.AreaSquare;
import RN.links.ELinkType;
import RN.links.Link;
import RN.nodes.ENodeType;

public class MatrixTest {

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testA() {
	}

	@Test
	public void test() {
	}
	
	@Test
	public void test2() {
	}
	
	@Test
	public void test3() {
	}
	
	@Test
	public void test4() {
		
		Double[][] m1 = new Double[10000][10000];
		Double[][] m2 = new Double[10000][10000];
		Double[][] m3 = new Double[10000][10000];
		Double[][] m4 = new Double[10000][10000];
		Double[][] m5 = new Double[10000][10000];
		
		System.out.println(m1[0][0]);
		
	}
	
	@Test
	public void test5() throws Exception{
		int width = 23;
		int centerIdx = (width -1) /2;
		int nbNode = width*width;
		AreaSquare area = new AreaSquare(nbNode);
		
		area.configureNode(false, ENodeType.REGULAR).createNodes(nbNode);
		for(int idx = 0; idx < nbNode; idx++){
			area.getNode(idx).addInput(Link.getInstance(ELinkType.REGULAR, true));
			area.getNode(idx).setEntry(0D);
		}
		
		for(int y = 0; y < width; y++){
			for(int x = 0; x < width; x++){
				if(y == centerIdx)
					area.getNodeXY(x , y).setEntry((double)x+y);
			}
		}
		
		System.out.print("Matrice : \n");
		for(int y = 0; y < width; y++){
			for(int x = 0; x < width; x++){
				System.out.print(area.getNodeXY(x , y).getInput(0).getValue());
				System.out.print("\t");
			}
			System.out.print("\n");
		}
		
		for(double theta = 0; theta <= 2*Math.PI; theta += Math.PI / 6D){
			System.out.println("\n Rotation d'angle theta = " + Math.toDegrees(theta) + " degree");
			for(int y = 0; y < width; y++){
				for(int x = 0; x < width; x++){
					double value = area.getNode(area.nodeXYToNodeId(x , y , centerIdx, centerIdx, theta)).getInput(0).getValue();
					System.out.print(value == 0D ? "." : value);
					System.out.print("\t");
				}
				System.out.print("\n");
			}
			System.out.print("\n\n");
			
		}
		
	}
	

	
	@Test
	public void test6(){
		for(double theta = 0; theta <= 2*Math.PI; theta += Math.PI / 2){
			System.out.println(Math.toDegrees(theta));
		}
		
		System.out.println();
		for(int y = 0; y < 3; y++){
			for(int x = 0; x < 3; x++){
				double thetaInit = Math.atan2(y - 1, x - 1);
				thetaInit = thetaInit < 0 ? thetaInit + 2*Math.PI : thetaInit;
				System.out.println(Math.toDegrees(thetaInit));
			}
		}
		
		System.out.println();
		int x0 = 3;
		int y0 = 2;
		int newX = 0;
		int newY = 0;
		for(int y = 0; y < 3; y++){
			for(int x = 0; x < 3; x++){
				
				double distance = Math.sqrt(Math.pow(x0 - (x-1 + x0), 2D) + Math.pow(y0 - (y-1 + y0), 2D));
				double thetaInit = Math.atan2(y-1, x-1);
				newX = ((int) (distance * (Math.cos(thetaInit) + 1) ) + x0);
				newY = ((int) (distance * (Math.sin(thetaInit) + 1) ) + y0);
				//System.out.println("angle=" + Math.toDegrees(thetaInit) + "  newX="+ newX + " newY=" + newY);
				System.out.print("\t");
			}
			System.out.println();
		}
	}
	
	@Test
	public void test7(){
		
		System.out.println(4%3);
		System.out.println(-12%10);
		
	}
	

}
