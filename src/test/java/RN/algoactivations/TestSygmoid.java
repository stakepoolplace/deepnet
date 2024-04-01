package RN.algoactivations;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import RN.algoactivations.SygmoidPerformer;

/**
 * @author Eric Marchand
 *
 */
public class TestSygmoid {

	@Test
	public void testPerform() throws Exception {
		SygmoidPerformer syg = new SygmoidPerformer();
		assertEquals(0.5D, syg.perform(0.0D), 0.1);
	}

	@Test
	public void testPerformDerivative() throws Exception {
		SygmoidPerformer syg = new SygmoidPerformer();
		assertEquals(0.25D, syg.performDerivative(0.0D), 0.1);
	}

}
