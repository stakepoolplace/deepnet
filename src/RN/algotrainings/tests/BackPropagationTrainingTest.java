package RN.algotrainings.tests;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import RN.INetwork;
import RN.ITester;
import RN.TestNetwork;
import RN.algotrainings.BackPropagationTrainer;
import RN.algotrainings.ITrainer;
import RN.dataset.inputsamples.InputSample;
import javafx.application.Application;
import javafx.stage.Stage;

/**
 * @author Eric Marchand
 *
 */
public class BackPropagationTrainingTest extends Application{

	public static ITester tester = null;
	public static ITrainer trainer = null;
	public static INetwork network = null;
	
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		
		Thread t = new Thread("JavaFX Init Thread") {
	        public void run() {
	        	
	        	Application.launch(BackPropagationTrainingTest.class, new String[0]);
	        	
	        }
	    };
	    t.setDaemon(true);
	    t.start();
	    Thread.sleep(1000);
		
		
		tester = TestNetwork.getInstance();
		trainer = new BackPropagationTrainer();
		InputSample.setFileSample(tester, tester.getFilePath(), 5);
		network = tester.getNetwork();
		System.out.println(network.getString());
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
	public final void testInitTrainer() {
		trainer.initTrainer();
	}

	@Test
	public final void testLaunchTrain() throws Exception {
		trainer.launchTrain();
		System.out.println(network.getString());
	}
	

	@Test
	public final void testTrain() throws Exception {
		trainer.train();
	}

	@Test
	public final void testNextTrainInputValues() {
		trainer.nextTrainInputValues();
	}

	@Test
	public final void testInitError() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testFeedForward() throws Exception {
		trainer.feedForward();
	}

	@Test
	public final void testGetLearningRate() {
		assertTrue(trainer.getLearningRate() == 0.5D);
	}

	@Test
	public final void testGetAlphaDeltaWeight() {
		assertTrue(trainer.getAlphaDeltaWeight() == 0.0D);
	}

	@Test
	public final void testGetErrorRate() {
		trainer.getErrorRate();
	}

	@Test
	public final void testSetErrorRate() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetLearningRate() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetAlphaDeltaWeight() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetDelay() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetDelta() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetNetwork() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetInputDataSetIterator() {
	}

	@Test
	public final void testSetInputDataSetIterator() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetLines() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetLines() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetCurrentEntry() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetCurrentEntry() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetCurrentOutputData() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetCurrentOutputData() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testIsBreakTraining() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetBreakTraining() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetAbsoluteError() {
		trainer.getAbsoluteError();
	}

	@Test
	public final void testSetAbsoluteError() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testGetMaxTrainingCycles() {
		fail("Not yet implemented"); // TODO
	}

	@Test
	public final void testSetMaxTrainingCycles() {
		fail("Not yet implemented"); // TODO
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		
	}

}
