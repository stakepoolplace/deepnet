package RN.algoactivations;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import RN.algoactivations.SygmoidPerformer2;

/**
 * @author Eric Marchand
 *
 */
public class TestSygmoid2 {

	@Test
	public void testPerform() throws Exception {
		SygmoidPerformer2 syg = new SygmoidPerformer2();
		assertEquals(0.0D, syg.perform(0.0D), 0D);
		assertEquals(0.5005D, syg.perform(0.1D), 0.00005D);
		assertEquals(-0.5005D, syg.perform(-0.1D), 0.00005D);
	}

	@Test
	public void testPerformDerivative() throws Exception {
		SygmoidPerformer2 syg = new SygmoidPerformer2();
		assertEquals(5.5D, syg.performDerivative(0.0D), 0D);
		assertEquals(0.01D, syg.performDerivative(0.7D), 0.0001D);
		assertEquals(0.01D, syg.performDerivative(-0.7D), 0.0001D);
	}

}
