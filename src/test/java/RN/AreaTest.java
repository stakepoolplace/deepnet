package RN;

import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.Area;
import RN.AreaSquare;
import RN.AreaSquareSampled;
import RN.IArea;
import RN.nodes.ENodeType;

public class AreaTest {
	
	private AreaSquare area = null;

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
	public void testArea() {
		int width = 3;
		int widthXheight = width * width;
		area = new AreaSquare(widthXheight);
		int x = 0; 
		int y = 0; 
		assertEquals(width * y, area.nodeXYToNodeId(0, y), 0D);
		x = 1; 
		assertEquals(1, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(1)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(1)[1], 0D);
		y = 1; 
		assertEquals(4, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(4)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(4)[1], 0D);
		y = 2; 
		assertEquals(7, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(7)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(7)[1], 0D);
		x = 2; 
		assertEquals(x + width * y, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(8, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(8)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(8)[1], 0D);
	}
	
	@Test
	public void testArea2() {
		int width = 100;
		int widthXheight = width * width;
		area = new AreaSquare(widthXheight);
		int x = 0; 
		int y = 0; 
		assertEquals(0, area.nodeXYToNodeId(0, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(0)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(0)[1], 0D);
		x = 99; 
		assertEquals(99, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(99)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(99)[1], 0D);
		y = 99; 
		assertEquals(9999, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(9999)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(9999)[1], 0D);

	}
	
	@Test
	public void testArea3() {
		int width = 100;
		int widthXheight = width * width;
		area = new AreaSquare(widthXheight);
		int x = 0; 
		int y = 0; 
		assertEquals(0, area.nodeXYToNodeId(0, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(0)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(0)[1], 0D);
		x = 99; 
		assertEquals(99, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(99)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(99)[1], 0D);
		y = 99; 
		assertEquals(9999, area.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, area.nodeIdToNodeXY(9999)[0], 0D);
		assertEquals(y, area.nodeIdToNodeXY(9999)[1], 0D);

	}
	
	@Test
	public void testArea4() {
		int width = 100;
		int widthXheight = width * width;
		area = new AreaSquare(widthXheight);
		area.configureNode(false, ENodeType.REGULAR).createNodes(widthXheight);
		
		assertEquals(widthXheight, area.getNodeCount(), 0D);
		
		AreaSquareSampled areaSampled = new AreaSquareSampled((AreaSquare) area, 2);
		
		assertEquals(Math.pow(width / 2, 2), areaSampled.getNodeCount(), 0D);
		
		int x = 0; 
		int y = 0; 
		assertEquals(0, areaSampled.nodeXYToNodeId(0, y), 0D);
		assertEquals(x, areaSampled.nodeIdToNodeXY(0)[0], 0D);
		assertEquals(y, areaSampled.nodeIdToNodeXY(0)[1], 0D);
		x = 49; 
		assertEquals(49, areaSampled.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, areaSampled.nodeIdToNodeXY(99)[0], 0D);
		assertEquals(y, areaSampled.nodeIdToNodeXY(49)[1], 0D);
		y = 49; 
		assertEquals(2499, areaSampled.nodeXYToNodeId(x, y), 0D);
		assertEquals(x, areaSampled.nodeIdToNodeXY(2499)[0], 0D);
		assertEquals(y, areaSampled.nodeIdToNodeXY(2499)[1], 0D);
		
		assertEquals(25, areaSampled.getNodeCenterX(), 0D);
		assertEquals(25, areaSampled.getNodeCenterY(), 0D);
		
		

	}
	
	@Test
	public void testArea5() {
		Sigma sigma = new Sigma(0D);
		sum(sigma);
		System.out.println(sigma.sig);
		
	}
	
	static class Sigma{
		public Double sig = null;
		public Sigma(Double sig){
			this.sig = sig;
		}
	}

	private void sum(Sigma sigma) {
		sigma.sig += 10;
	}
	

}
