package RN.tests;

import static org.ojalgo.constant.PrimitiveMath.PI;
import static org.ojalgo.constant.PrimitiveMath.ZERO;
import static org.ojalgo.function.PrimitiveFunction.DIVIDE;
import static org.ojalgo.function.PrimitiveFunction.SUBTRACT;

import java.io.File;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.ojalgo.OjAlgoUtils;
import org.ojalgo.array.Array2D;
import org.ojalgo.array.ArrayAnyD;
import org.ojalgo.array.BufferArray;
import org.ojalgo.function.aggregator.AggregatorFunction;
import org.ojalgo.function.aggregator.PrimitiveAggregator;
import org.ojalgo.machine.JavaType;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.netio.BasicLogger;
import org.ojalgo.random.Uniform;

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
		
        BasicLogger.debug();
        BasicLogger.debug(MatrixTest.class.getSimpleName());
        BasicLogger.debug(OjAlgoUtils.getTitle());
        BasicLogger.debug(OjAlgoUtils.getDate());
        BasicLogger.debug();

        // The file pathname - previously existing or not
        final File tmpFile = new File("BasicDemo.array");

        final long tmpJvmMemory = OjAlgoUtils.ENVIRONMENT.memory;
        BasicLogger.debug("The JVM was started with a max heap size of {}MB", (tmpJvmMemory / 1024L) / 1024L);

        final long tmpMaxDimension = (long) Math.sqrt(tmpJvmMemory / JavaType.DOUBLE.memory());
        BasicLogger.debug("The maximum number of rows and columns: {}", tmpMaxDimension);
        // Disregarding any overhead and all other objects

        // A max sized 2-dimensional file based array
        final Array2D<Double> tmpArray2D = BufferArray.make(tmpFile, tmpMaxDimension, tmpMaxDimension);

        // An equally sized multi/any-dimensional array, based on the same file
        final ArrayAnyD<Double> tmpArrayAnyD = BufferArray.make(tmpFile, tmpMaxDimension, tmpMaxDimension, 1L, 1L);
        // An any-dimensional array can of course be 1- or 2-dimensional. In this case we instantiated a 4-dimensional
        // array, but the size of the 3:d and 4:th dimensions are just 1. Effectively this is just a 2-dimensioanl array.

        // Fill the entire array/file with zeros
        tmpArrayAnyD.fillAll(ZERO);

        BasicLogger.debug("Number of elements in...");
        BasicLogger.debug("\t2D: {}", tmpArray2D.count());
        BasicLogger.debug("\tAnyD: {}", tmpArrayAnyD.count());

        final AggregatorFunction<Double> tmpCardinality = PrimitiveAggregator.getSet().cardinality();

        final long tmpRowIndex = Uniform.randomInteger(tmpMaxDimension);
        final long tmpColumnIndex = Uniform.randomInteger(tmpMaxDimension);

        // Using the arbitrary dimensinal interface/facade we will update an entire row (all columns) of the first matrix of the first cube...
        long[] tmpReferenceToFirstElement = new long[] { tmpRowIndex, 0L, 0L, 0L };
        int tmpDimension = 1; // That's the column-dimension
        tmpArrayAnyD.fillSet(tmpReferenceToFirstElement, tmpDimension, PI);

        // Using the arbitrary dimensional interface/facade we will update an entire row (all columns) of the first matrix of the first cube...
        tmpReferenceToFirstElement = new long[] { 0L, tmpColumnIndex, 0L, 0L };
        tmpDimension = 0; // That's the row-dimension
        tmpArrayAnyD.fillSet(tmpReferenceToFirstElement, tmpDimension, PI);

        // So far we've been writing to the array-file using ArrayAnyD
        // Now we'll switch to using Array2D, but they're both mapped to the same files

        tmpCardinality.reset();
        tmpArray2D.visitRow(tmpRowIndex, 0L, tmpCardinality);
        BasicLogger.debug("Number of nonzero elements in row {}: {}", tmpRowIndex, tmpCardinality.intValue());

        tmpCardinality.reset();
        tmpArray2D.visitColumn(0L, tmpColumnIndex, tmpCardinality);
        BasicLogger.debug("Number of nonzero elements in column {}: {}", tmpColumnIndex, tmpCardinality.intValue());

        tmpCardinality.reset();
        tmpArray2D.visitAll(tmpCardinality);
        BasicLogger.debug("Number of nonzero elements in the 2D array: {}", tmpCardinality.intValue());

        BasicLogger.debug("Divide the elements of row {} to create 1.0:s", tmpRowIndex);
        tmpArray2D.modifyRow(tmpRowIndex, 0L, DIVIDE.second(PI));
        BasicLogger.debug("Subtract from the elements of column {} to create 0.0:s", tmpColumnIndex);
        tmpArray2D.modifyColumn(0L, tmpColumnIndex, SUBTRACT.second(PI));
        BasicLogger.debug("Explictly set the intersection element to 0.0 using the arbitrary-dimensional array.");
        tmpArrayAnyD.set(new long[] { tmpRowIndex, tmpColumnIndex }, ZERO);

        final AggregatorFunction<Double> tmpSum = PrimitiveAggregator.getSet().sum();
        BasicLogger.debug("Expected sum of all elements: {}", tmpMaxDimension - 1L);
        tmpSum.reset();
        tmpArray2D.visitAll(tmpSum);
        BasicLogger.debug("Actual sum of all elements: {}", tmpSum.intValue());

	}

	@Test
	public void test() {
		
		final PhysicalStore.Factory<Double, PrimitiveDenseStore> tmpFactory = PrimitiveDenseStore.FACTORY;
		
		final PrimitiveDenseStore pds = tmpFactory.rows(new double[]{1D, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D});
		
		MatrixStore<Double> ms = pds.builder().row(1,9).build();
		System.out.println(ms.toString());
		
	}
	
	@Test
	public void test2() {
		
//		DoubleMatrix values = new DoubleMatrix(new double[][]{{1,2,3,4,5,6,7,8,9}});
//		System.out.println(values.toString("%.2f", "", "", "\t", "\n"));
//		
//		
//		System.out.println();
//		DoubleMatrix weights = new DoubleMatrix(9,2);
//		weights.putColumn(0, new DoubleMatrix(new double[]{1,2,3,4,5,6,7,8,9}));
//		weights.putColumn(1, new DoubleMatrix(new double[]{1,2,3,4,5,6,7,8,9}));
//		System.out.println(weights.toString("%.2f", "", "", "\t", "\n"));
//		
//		
//		System.out.println();
//		//DoubleMatrix results = values.mul(weights);
//		DoubleMatrix results2 = values.mmul(weights);
//		//System.out.println(results.toString(" %.2f ", "", "", "\t", "\n"));
//		System.out.println(results2.toString(" %.2f ", "", "", "\t", "\n"));
		
		
	}
	
	@Test
	public void test3() {
		
		final PhysicalStore.Factory<Double, PrimitiveDenseStore> tmpFactory = PrimitiveDenseStore.FACTORY;
		
		final PrimitiveDenseStore pds = tmpFactory.makeZero(10000, 10000);
		final PrimitiveDenseStore pds2 = tmpFactory.makeZero(10000, 10000);
		final PrimitiveDenseStore pds3 = tmpFactory.makeZero(10000, 10000);
		final PrimitiveDenseStore pds4 = tmpFactory.makeZero(10000, 10000);
		
		MatrixStore<Double> ms = pds.builder().build();
		System.out.println(ms.toString());
		
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
