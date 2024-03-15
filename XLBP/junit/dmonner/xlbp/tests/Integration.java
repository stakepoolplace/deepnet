package dmonner.xlbp.tests;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Test;

import dmonner.xlbp.Component;
import dmonner.xlbp.Network;
import dmonner.xlbp.SetWeightInitializer;
import dmonner.xlbp.UniformWeightInitializer;
import dmonner.xlbp.WeightUpdaterType;
import dmonner.xlbp.compound.FunctionCompound;
import dmonner.xlbp.compound.InputCompound;
import dmonner.xlbp.compound.LinearTargetCompound;
import dmonner.xlbp.compound.MemoryCellCompound;
import dmonner.xlbp.connection.Connection;
import dmonner.xlbp.layer.Layer;
import dmonner.xlbp.trial.Step;
import dmonner.xlbp.trial.Trial;

public class Integration
{
	public static void assertEqualTolerance(final float target, final float actual)
	{
		final int float_max_precision = 6;
		final float min_positive_float = (float) Math.pow(10, -float_max_precision);

		// determine the magnitude of the target via log_10
		final int magnitude = 1 + (Math.abs(target) < min_positive_float ? -1 : (int) Math.floor(Math
				.log10(Math.abs(target))));

		// the precision we can expect is equal to the float's max precision minus the magnitude
		final int precision = float_max_precision - Math.max(0, magnitude);

		// the tolerance (error bounding term) is calculated based on precision
		final float tolerance = (float) Math.pow(10, -precision);

		// do the actual comparison with tolerance
		if(Math.abs(target - actual) >= tolerance)
			// NOTE: SET BREAKPOINT HERE TO DEBUG A COMPARISON FAILURE!!
			fail("target " + target + " != " + actual + " actual (tolerance = " + tolerance + ")");
	}

	private final int in = 0;
	private final int ig1 = 1;
	private final int fg1 = 2;
	private final int in1 = 3;
	private final int st1 = 4;
	private final int mc1 = 5;
	private final int og1 = 6;
	private final int mg1 = 7;
	private final int ig2 = 8;
	private final int fg2 = 9;
	private final int in2 = 10;
	private final int st2 = 11;
	private final int mc2 = 12;
	private final int og2 = 13;
	private final int mg2 = 14;
	private final int out = 15;
	private final int nlayers = 16;

	private final int mc1f = 0;
	private final int mc1b = 1;
	private final int ig1b = 2;
	private final int ig1p = 3;
	private final int ig1l = 4;
	private final int ig1f = 5;
	private final int fg1b = 6;
	private final int fg1p = 7;
	private final int fg1l = 8;
	private final int fg1f = 9;
	private final int og1b = 10;
	private final int og1p = 11;
	private final int og1l = 12;
	private final int og1f = 13;
	private final int mc2f = 14;
	private final int mc2b = 15;
	private final int ig2b = 16;
	private final int ig2p = 17;
	private final int ig2l = 18;
	private final int ig2f = 19;
	private final int fg2b = 20;
	private final int fg2p = 21;
	private final int fg2l = 22;
	private final int fg2f = 23;
	private final int og2b = 24;
	private final int og2p = 25;
	private final int og2l = 26;
	private final int og2f = 27;
	private final int outb = 28;
	private final int outf = 29;
	private final int nweights = 30;

	private String function;
	private int nsteps;
	private int size;
	private float[][][][] w;
	private float[][][][] e;
	private float[][][] a;
	private float[][][] r;
	private float[][][] d;
	private float[][] tg;
	private Network net;

	private boolean ig;
	private boolean fg;
	private boolean og;
	private boolean ph;
	private boolean gl;
	private boolean ul;
	private boolean tr;

	private void add(final float[] dest, final float[] src)
	{
		assertTrue(dest.length == src.length);
		for(int j = 0; j < dest.length; j++)
			dest[j] += src[j];
	}

	private void addb(final float[] dest, final float[][] b)
	{
		assertTrue(dest.length == b.length);
		for(int j = 0; j < dest.length; j++)
		{
			assertTrue(dest.length == b[j].length);
			dest[j] += b[j][j];
		}
	}

	private void adde(final float[][] elig, final float[] in)
	{
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == in.length);
			for(int i = 0; i < in.length; i++)
				elig[j][i] += in[i];
		}
	}

	private void addeb(final float[][] elig)
	{
		for(int j = 0; j < elig.length; j++)
			elig[j][j] += 1F;
	}

	private void addebg(final float[][] elig, final float[] gate)
	{
		assertTrue(elig.length == gate.length);
		for(int j = 0; j < elig.length; j++)
			elig[j][j] += gate[j];
	}

	private void added(final float[][] elig, final float[] in)
	{
		assertTrue(elig.length == in.length);
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == in.length);
			elig[j][j] += in[j];
		}
	}

	private void addeg(final float[][] elig, final float[] in, final float[] gate)
	{
		assertTrue(elig.length == gate.length);
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == in.length);
			for(int i = 0; i < in.length; i++)
				elig[j][i] += in[i] * gate[j];
		}
	}

	private void addegd(final float[][] elig, final float[] in, final float[] gate)
	{
		assertTrue(elig.length == gate.length);
		assertTrue(elig.length == in.length);
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == in.length);
			elig[j][j] += in[j] * gate[j];
		}
	}

	private void addep(final float[][] elig, final float[][] prev)
	{
		assertTrue(elig.length == prev.length);
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == prev[j].length);
			for(int i = 0; i < elig[j].length; i++)
				elig[j][i] += prev[j][i];
		}
	}

	private void addepg(final float[][] elig, final float[][] prev, final float[] gate)
	{
		assertTrue(elig.length == prev.length);
		for(int j = 0; j < elig.length; j++)
		{
			assertTrue(elig[j].length == gate.length);
			assertTrue(elig[j].length == prev[j].length);
			for(int i = 0; i < gate.length; i++)
				elig[j][i] += prev[j][i] * gate[j];
		}
	}

	private void addg(final float[] dest, final float[] src, final float[] gate)
	{
		assertTrue(dest.length == src.length);
		assertTrue(dest.length == gate.length);
		for(int j = 0; j < dest.length; j++)
			dest[j] += src[j] * gate[j];
	}

	private void addw(final float[] dest, final float[] src, final float[][] w)
	{
		assertTrue(dest.length == w.length);
		for(int j = 0; j < dest.length; j++)
		{
			assertTrue(src.length == w[j].length);
			for(int i = 0; i < src.length; i++)
				dest[j] += w[j][i] * src[i];
		}
	}

	private void backpropm(final float[] dfrom, final float[] dto, final float[] m)
	{
		assertTrue(m.length == dto.length);
		assertTrue(m.length == dfrom.length);
		for(int j = 0; j < m.length; j++)
			dfrom[j] += m[j] * dto[j];
	}

	private void backpropw(final float[] dfrom, final float[] dto, final float[][] w)
	{
		assertTrue(w.length == dto.length);
		for(int j = 0; j < w.length; j++)
		{
			assertTrue(w[j].length == dfrom.length);
			for(int i = 0; i < w[j].length; i++)
				dfrom[i] += w[j][i] * dto[j];
		}
	}

	private void backpropwr(final float[] dfrom, final float[] dto, final float[] rto,
			final float[][] w)
	{
		assertTrue(w.length == dto.length);
		assertTrue(w.length == rto.length);
		for(int j = 0; j < w.length; j++)
		{
			assertTrue(w[j].length == dfrom.length);
			for(int i = 0; i < w[j].length; i++)
				dfrom[i] += w[j][i] * dto[j] * rto[j];
		}
	}

	private Network buildNetwork(final String mctype, final String function)
	{
		final Network net = new Network("Network");
		net.setWeightUpdaterType(WeightUpdaterType.basic(1F));

		final InputCompound in = new InputCompound("In", 2);
		final MemoryCellCompound mc1 = new MemoryCellCompound("MC1", 2, mctype, function);
		final MemoryCellCompound mc2 = new MemoryCellCompound("MC2", 2, mctype, function);
		final LinearTargetCompound out = new LinearTargetCompound("Out", 2);

		mc1.addUpstreamWeights(in);
		mc2.addUpstreamWeights(mc1);
		out.addUpstreamWeights(mc2);

		net.add(in);
		net.add(mc1);
		net.add(mc2);
		net.add(out);

		return net;
	}

	private int compareActivations(final Component comp, final float[][] a, final int idx)
	{
		if(comp instanceof InputCompound)
			return compareActivations((InputCompound) comp, a, idx);
		else if(comp instanceof MemoryCellCompound)
			return compareActivations((MemoryCellCompound) comp, a, idx);
		else if(comp instanceof FunctionCompound)
			return compareActivations((FunctionCompound) comp, a, idx);
		else
			throw new IllegalArgumentException("Unhandled Component type: " + comp.getClass());
	}

	private int compareActivations(final FunctionCompound fc, final float[][] a, int idx)
	{
		compareActivations(fc.getActLayer(), a, idx++);

		return idx;
	}

	private int compareActivations(final InputCompound ic, final float[][] a, int idx)
	{
		compareActivations(ic.getInputLayer(), a, idx++);

		return idx;
	}

	private void compareActivations(final Layer layer, final float[][] a, final int idx)
	{
		final float[] target = a[idx];
		final float[] actual = layer.getActivations();

		assertTrue(target.length == actual.length);

		for(int i = 0; i < target.length; i++)
			assertEqualTolerance(target[i], actual[i]);
	}

	private int compareActivations(final MemoryCellCompound mc, final float[][] a, int idx)
	{
		if(ig)
			compareActivations(mc.getInputGates(), a, idx);
		idx++;

		if(fg)
			compareActivations(mc.getForgetGates(), a, idx);
		idx++;

		compareActivations(mc.getNetInputLayer(), a, idx++);
		compareActivations(mc.getStateLayer(), a, idx++);
		idx = compareActivations(mc.getMemoryCells(), a, idx);

		if(og)
			compareActivations(mc.getOutputGates(), a, idx);
		idx++;

		compareActivations(mc.getOutput(), a, idx++);

		return idx;
	}

	private void compareActivations(final Network net, final float[][] a)
	{
		int idx = 0;
		for(int i = 0; i < net.size(); i++)
			idx = compareActivations(net.getComponent(i), a, idx);

		if(idx < a.length)
			throw new IllegalStateException("Only used " + idx + " out of " + a.length
					+ " weight matrices!");
	}

	private int compareEligibilities(final Component comp, final float[][][] e, final int idx)
	{
		if(comp instanceof InputCompound)
			return idx; // do nothing!
		else if(comp instanceof MemoryCellCompound)
			return compareEligibilities((MemoryCellCompound) comp, e, idx);
		else if(comp instanceof FunctionCompound)
			return compareEligibilities((FunctionCompound) comp, e, idx);
		else
			throw new IllegalArgumentException("Unhandled Component type: " + comp.getClass());
	}

	private void compareEligibilities(final Connection conn, final float[][][] e, final int idx)
	{
		final float[][] target = e[idx];
		final float[][] actual = conn.toEligibilitiesMatrix();

		assertTrue(target.length == actual.length);

		for(int i = 0; i < target.length; i++)
		{
			assertTrue(target[i].length == actual[i].length);

			for(int j = 0; j < target[i].length; j++)
				assertEqualTolerance(target[i][j], actual[i][j]);
		}
	}

	private int compareEligibilities(final FunctionCompound tc, final float[][][] e, int idx)
	{
		compareEligibilities(tc.getBiasInput().getConnection(), e, idx++);
		for(int i = 0; i < tc.nUpstreamWeights(); i++)
			compareEligibilities(tc.getUpstreamWeights(i).getConnection(), e, idx++);

		return idx;
	}

	private int compareEligibilities(final MemoryCellCompound mc, final float[][][] e, int idx)
	{
		for(int i = 0; i < mc.nUpstreamWeights(); i++)
			compareEligibilities(mc.getUpstreamWeights(i).getConnection(), e, idx++);

		idx = compareEligibilities(mc.getMemoryCells(), e, idx);

		if(ig)
			compareGateEligibilities(mc.getInputGates(), e, idx);
		idx += 4;

		if(fg)
			compareGateEligibilities(mc.getForgetGates(), e, idx);
		idx += 4;

		if(og)
			compareGateEligibilities(mc.getOutputGates(), e, idx);
		idx += 4;

		return idx;
	}

	private void compareEligibilities(final Network net, final float[][][] e)
	{
		int idx = 0;
		for(int i = 0; i < net.size(); i++)
			idx = compareEligibilities(net.getComponent(i), e, idx);

		if(idx < e.length)
			throw new IllegalStateException("Only used " + idx + " out of " + e.length
					+ " weight matrices!");
	}

	private void compareGateEligibilities(final FunctionCompound gate, final float[][][] e, int idx)
	{
		int nw = 0;

		// b
		compareEligibilities(gate.getBiasInput().getConnection(), e, idx);
		idx++;

		// p
		if(ph)
			compareEligibilities(gate.getUpstreamWeights(nw++).getConnection(), e, idx);
		idx++;

		// l
		if(gl || ul)
			compareEligibilities(gate.getUpstreamWeights(nw++).getConnection(), e, idx);
		idx++;

		// f
		compareEligibilities(gate.getUpstreamWeights(nw++).getConnection(), e, idx);
		idx++;

		assertTrue(gate.nUpstreamWeights() == nw);
	}

	private void compareGateWeights(final FunctionCompound gate, final float[][][] w, int idx)
	{
		int nw = 1;
		if(ph)
			nw++;
		if(gl || ul)
			nw++;

		assertTrue(gate.nUpstreamWeights() == nw);

		// f
		compareWeights(gate.getUpstreamWeights(--nw).getConnection(), w, idx);
		idx--;

		// l
		if(gl || ul)
			compareWeights(gate.getUpstreamWeights(--nw).getConnection(), w, idx);
		idx--;

		// p
		if(ph)
			compareWeights(gate.getUpstreamWeights(--nw).getConnection(), w, idx);
		idx--;

		// b
		compareWeights(gate.getBiasInput().getConnection(), w, idx);
		idx--;
	}

	private int compareResponsibilities(final Component comp, final float[][] d, final float[][] r,
			final int idx)
	{
		if(comp instanceof InputCompound)
			return compareResponsibilities((InputCompound) comp, d, idx);
		else if(comp instanceof MemoryCellCompound)
			return compareResponsibilities((MemoryCellCompound) comp, d, r, idx);
		else if(comp instanceof FunctionCompound)
			return compareResponsibilities((FunctionCompound) comp, d, idx);
		else
			throw new IllegalArgumentException("Unhandled Component type: " + comp.getClass());
	}

	private int compareResponsibilities(final FunctionCompound fc, final float[][] d, int idx)
	{
		compareResponsibilities(fc.getNetLayer(), d, idx--);

		return idx;
	}

	private int compareResponsibilities(final InputCompound ic, final float[][] d, int idx)
	{
		compareResponsibilities(ic.getInputLayer(), d, idx--);

		return idx;
	}

	private void compareResponsibilities(final Layer layer, final float[][] d, final int idx)
	{
		final float[] target = d[idx];
		final float[] actual = layer.getResponsibilities().get();

		assertTrue(target.length == actual.length);

		for(int i = 0; i < target.length; i++)
			assertEqualTolerance(target[i], actual[i]);
	}

	private int compareResponsibilities(final MemoryCellCompound mc, final float[][] d,
			final float[][] r, int idx)
	{
		// everything after copy-point gets compared to d
		compareResponsibilities(mc.getOutput(), d, idx--);
		if(og)
			compareResponsibilities(mc.getOutputGates(), d, idx);
		idx--;
		idx = compareResponsibilities(mc.getMemoryCells(), d, idx);
		compareResponsibilities(mc.getStateLayer(), d, idx--);

		// everything before to the copy-point gets compared to r
		compareResponsibilities(mc.getNetInputLayer(), r, idx--);
		if(fg)
			compareResponsibilities(mc.getForgetGates(), r, idx);
		idx--;
		if(ig)
			compareResponsibilities(mc.getInputGates(), r, idx);
		idx--;

		return idx;
	}

	private void compareResponsibilities(final Network net, final float[][] d, final float[][] r)
	{
		int idx = nlayers - 1;
		for(int i = net.size() - 1; i >= 0; i--)
			idx = compareResponsibilities(net.getComponent(i), d, r, idx);

		if(idx >= 0)
			throw new IllegalStateException("Only used " + (nlayers - idx) + " out of " + d.length
					+ " weight matrices!");
	}

	private int compareWeights(final Component comp, final float[][][] w, final int idx)
	{
		if(comp instanceof InputCompound)
			return idx; // do nothing!
		else if(comp instanceof MemoryCellCompound)
			return compareWeights((MemoryCellCompound) comp, w, idx);
		else if(comp instanceof FunctionCompound)
			return compareWeights((FunctionCompound) comp, w, idx);
		else
			throw new IllegalArgumentException("Unhandled Component type: " + comp.getClass());
	}

	private void compareWeights(final Connection conn, final float[][][] w, final int idx)
	{
		final float[][] target = w[idx];
		final float[][] actual = conn.toMatrix();

		assertTrue(target.length == actual.length);

		for(int i = 0; i < target.length; i++)
		{
			assertTrue(target[i].length == actual[i].length);

			for(int j = 0; j < target[i].length; j++)
				assertEqualTolerance(target[i][j], actual[i][j]);
		}
	}

	private int compareWeights(final FunctionCompound tc, final float[][][] w, int idx)
	{
		for(int i = tc.nUpstreamWeights() - 1; i >= 0; i--)
			compareWeights(tc.getUpstreamWeights(i).getConnection(), w, idx--);
		compareWeights(tc.getBiasInput().getConnection(), w, idx--);

		return idx;
	}

	private int compareWeights(final MemoryCellCompound mc, final float[][][] w, int idx)
	{
		if(og)
			compareGateWeights(mc.getOutputGates(), w, idx);
		idx -= 4;

		if(fg)
			compareGateWeights(mc.getForgetGates(), w, idx);
		idx -= 4;

		if(ig)
			compareGateWeights(mc.getInputGates(), w, idx);
		idx -= 4;

		idx = compareWeights(mc.getMemoryCells(), w, idx);

		assertTrue(mc.nUpstreamWeights() == 1);
		compareWeights(mc.getUpstreamWeights(0).getConnection(), w, idx--);

		return idx;
	}

	private void compareWeights(final Network net, final float[][][] w)
	{
		int idx = nweights - 1;
		for(int i = net.size() - 1; i >= 0; i--)
			idx = compareWeights(net.getComponent(i), w, idx);

		if(idx >= 0)
			throw new IllegalStateException("Only used " + (nweights - idx) + " out of " + w.length
					+ " weight matrices!");
	}

	private void copy(final float[] dest, final float[] src)
	{
		assertTrue(dest.length == src.length);
		System.arraycopy(src, 0, dest, 0, dest.length);
	}

	private void copyWeights(final float[][][][] w, final int tFrom, final int tTo)
	{
		assertTrue(w[tFrom].length == w[tTo].length);
		for(int i = 0; i < w[tFrom].length; i++)
		{
			assertTrue(w[tFrom][i].length == w[tTo][i].length);
			for(int j = 0; j < w[tFrom][i].length; j++)
			{
				assertTrue(w[tFrom][i][j].length == w[tTo][i][j].length);
				for(int k = 0; k < w[tFrom][i][j].length; k++)
					w[tTo][i][j][k] = w[tFrom][i][j][k];
			}
		}
	}

	private float f(final float net)
	{
		return 1F / (1F + (float) Math.exp(-net));
	}

	private void f(final float[] act, final float[] net)
	{
		assertTrue(act.length == net.length);
		for(int j = 0; j < act.length; j++)
			act[j] = f(net[j]);
	}

	private float fp(final float act)
	{
		return act * (1 - act);
	}

	private void fp(final float[] fp, final float[] act)
	{
		assertTrue(fp.length == act.length);
		for(int j = 0; j < fp.length; j++)
			fp[j] = fp(act[j]);
	}

	private void mul(final float[] dest, final float[] a, final float[] b)
	{
		assertTrue(dest.length == a.length);
		assertTrue(dest.length == b.length);
		for(int i = 0; i < dest.length; i++)
			dest[i] = a[i] * b[i];
	}

	private void ones(final float[] dest)
	{
		for(int j = 0; j < dest.length; j++)
			dest[j] = 1F;
	}

	private void processTrialStep(final Step step, final int i)
	{
		net.clearInputs();
		step.train();

		xforward(i);
		if(step.nTargets() > 0)
			xbackward(i);
		else
			copyWeights(w, i - 1, i);

		compareActivations(net, a[i]);
		compareEligibilities(net, e[i]);
		compareResponsibilities(net, d[i], r[i]);
		compareWeights(net, w[i]);
	}

	private void setGateWeights(final FunctionCompound gate, final float[][][] w, int idx)
	{
		int nw = 0;

		// b
		setWeights(gate.getBiasInput().getConnection(), w, idx);
		idx++;

		// p
		if(ph)
			setWeights(gate.getUpstreamWeights(nw++).getConnection(), w, idx);
		idx++;

		// l
		if(gl || ul)
			setWeights(gate.getUpstreamWeights(nw++).getConnection(), w, idx);
		idx++;

		// f
		setWeights(gate.getUpstreamWeights(nw++).getConnection(), w, idx);
		idx++;

		assertTrue(gate.nUpstreamWeights() == nw);
	}

	private int setWeights(final Component comp, final float[][][] w, final int idx)
	{
		if(comp instanceof InputCompound)
			return idx; // do nothing!
		else if(comp instanceof MemoryCellCompound)
			return setWeights((MemoryCellCompound) comp, w, idx);
		else if(comp instanceof FunctionCompound)
			return setWeights((FunctionCompound) comp, w, idx);
		else
			throw new IllegalArgumentException("Unhandled Component type: " + comp.getClass());
	}

	private void setWeights(final Connection conn, final float[][][] w, final int idx)
	{
		conn.setWeightInitializer(new SetWeightInitializer(w[idx], true));
	}

	private int setWeights(final FunctionCompound fc, final float[][][] w, int idx)
	{
		setWeights(fc.getBiasInput().getConnection(), w, idx++);
		for(int i = 0; i < fc.nUpstreamWeights(); i++)
			setWeights(fc.getUpstreamWeights(i).getConnection(), w, idx++);

		return idx;
	}

	private int setWeights(final MemoryCellCompound mc, final float[][][] w, int idx)
	{
		for(int i = 0; i < mc.nUpstreamWeights(); i++)
			setWeights(mc.getUpstreamWeights(i).getConnection(), w, idx++);

		idx = setWeights(mc.getMemoryCells(), w, idx);

		if(ig)
			setGateWeights(mc.getInputGates(), w, idx);
		idx += 4;

		if(fg)
			setGateWeights(mc.getForgetGates(), w, idx);
		idx += 4;

		if(og)
			setGateWeights(mc.getOutputGates(), w, idx);
		idx += 4;

		return idx;
	}

	private void setWeights(final Network net, final float[][][] w)
	{
		int idx = 0;
		for(int i = 0; i < net.size(); i++)
			idx = setWeights(net.getComponent(i), w, idx);

		if(idx < w.length)
			throw new IllegalStateException("Only used " + idx + " out of " + w.length
					+ " weight matrices!");
	}

	@Test
	public void test()
	{
		test("", true);
		test("", false);
	}

	private void test(final String mctype, final boolean adjlist)
	{
		ig = mctype.contains("I");
		fg = mctype.contains("F");
		og = mctype.contains("O");
		ph = mctype.contains("P");
		gl = mctype.contains("L") || mctype.contains("G");
		ul = mctype.contains("U");
		tr = mctype.contains("T");

		if(ul && (ph || gl))
			fail("Unsupported configuration: " + mctype);

		function = "logistic";
		nsteps = 6;
		size = 2;

		// static initial weights
		final float[][][] w0 = new float[][][] { //
		// /// mc1
				{ { 0.11F, 0.12F }, { 0.13F, 0.14F } }, //
				// mc1mc
				{ { 0.15F, 0.00F }, { 0.00F, 0.16F } }, //
				// mc1ig
				{ { 0.17F, 0.00F }, { 0.00F, 0.18F } }, //
				{ { 0.23F, 0.00F }, { 0.00F, 0.24F } }, //
				{ { 0.25F, 0.26F }, { 0.26F, 0.28F } }, //
				{ { 0.19F, 0.20F }, { 0.21F, 0.22F } }, //
				// mc1fg
				{ { 0.29F, 0.00F }, { 0.00F, 0.30F } }, //
				{ { 0.35F, 0.00F }, { 0.00F, 0.36F } }, //
				{ { 0.37F, 0.38F }, { 0.39F, 0.40F } }, //
				{ { 0.31F, 0.32F }, { 0.33F, 0.34F } }, //
				// mc1og
				{ { 0.41F, 0.00F }, { 0.00F, 0.42F } }, //
				{ { 0.47F, 0.00F }, { 0.00F, 0.48F } }, //
				{ { 0.49F, 0.50F }, { 0.51F, 0.52F } }, //
				{ { 0.43F, 0.44F }, { 0.45F, 0.46F } }, //
				// mc2
				{ { 0.53F, 0.54F }, { 0.55F, 0.56F } }, //
				// mc2mc
				{ { 0.57F, 0.00F }, { 0.00F, 0.58F } }, //
				// mc2ig
				{ { 0.59F, 0.00F }, { 0.00F, 0.60F } }, //
				{ { 0.65F, 0.00F }, { 0.00F, 0.66F } }, //
				{ { 0.67F, 0.68F }, { 0.69F, 0.70F } }, //
				{ { 0.61F, 0.62F }, { 0.63F, 0.64F } }, //
				// mc2fg
				{ { 0.71F, 0.00F }, { 0.00F, 0.72F } }, //
				{ { 0.77F, 0.00F }, { 0.00F, 0.78F } }, //
				{ { 0.79F, 0.80F }, { 0.81F, 0.82F } }, //
				{ { 0.73F, 0.74F }, { 0.75F, 0.76F } }, //
				// mc2og
				{ { 0.83F, 0.00F }, { 0.00F, 0.84F } }, //
				{ { 0.89F, 0.00F }, { 0.00F, 0.90F } }, //
				{ { 0.91F, 0.92F }, { 0.93F, 0.94F } }, //
				{ { 0.85F, 0.86F }, { 0.87F, 0.88F } }, //
				// out
				{ { 0.95F, 0.00F }, { 0.00F, 0.96F } }, //
				{ { 0.97F, 0.98F }, { 0.99F, 1.00F } }, //
		};

		// set up buffers, inputs, and targets
		w = new float[nsteps][nweights][size][size];
		e = new float[nsteps][nweights][size][size];
		w[0] = w0;
		a = new float[nsteps][nlayers][size];
		r = new float[nsteps][nlayers][size];
		d = new float[nsteps][nlayers][size];
		a[1][0] = new float[] { 0F, 1F };
		a[2][0] = new float[] { 1F, 0F };
		a[3][0] = new float[] { 1F, 1F };
		a[4][0] = new float[] { 0F, 0F };
		a[5][0] = new float[] { 1F, 1F };
		tg = new float[nsteps][];
		tg[2] = new float[] { 0F, 1F };
		tg[3] = new float[] { 1F, 0F };
		tg[5] = new float[] { 1F, 1F };

		// build network
		net = buildNetwork(mctype, function);

		if(adjlist) // hack to use AdjacencyListConnection where possible
		{
			// prevent warning messages
			((MemoryCellCompound) net.getComponent(1)).setPeepholeFullOnly(false);
			((MemoryCellCompound) net.getComponent(2)).setPeepholeFullOnly(false);
			// set to a weight density <1 to ensure AdjacencyList
			net.setWeightInitializer(new UniformWeightInitializer(0.5F));
		}

		setWeights(net, w[0]);
		net.optimize();
		net.build();
		net.clear();

		// make sure network is empty and weights are as set
		compareActivations(net, a[0]);
		compareEligibilities(net, e[0]);
		compareResponsibilities(net, d[0], r[0]);
		compareWeights(net, w[0]);

		// create a training trial
		final Trial trial = new Trial(net);
		for(int i = 1; i < nsteps; i++)
		{
			final Step step = trial.nextStep();
			step.addInput(a[i][0]);
			if(tg[i] != null)
				step.addTarget(tg[i]);
		}

		// run the trial & test correctness along the way
		final List<Step> steps = trial.getSteps();
		for(int i = 1; i < nsteps; i++)
			processTrialStep(steps.get(i - 1), i);

		// clear the network and test return to base state
		net.clear();
		compareActivations(net, a[0]);
		compareEligibilities(net, e[0]);
		compareResponsibilities(net, d[0], r[0]);
		compareWeights(net, w[nsteps - 1]);
	}

	@Test
	public void testF()
	{
		test("F", true);
		test("F", false);
	}

	@Test
	public void testFL()
	{
		test("FL", true);
		test("FL", false);
	}

	@Test
	public void testFLT()
	{
		test("FLT", true);
		test("FLT", false);
	}

	@Test
	public void testFO()
	{
		test("FO", true);
		test("FO", false);
	}

	@Test
	public void testFOL()
	{
		test("FOL", true);
		test("FOL", false);
	}

	@Test
	public void testFOLT()
	{
		test("FOLT", true);
		test("FOLT", false);
	}

	@Test
	public void testFOP()
	{
		test("FOP", true);
		test("FOP", false);
	}

	@Test
	public void testFOPL()
	{
		test("FOPL", true);
		test("FOPL", false);
	}

	@Test
	public void testFOPLT()
	{
		test("FOPLT", true);
		test("FOPLT", false);
	}

	@Test
	public void testFOPT()
	{
		test("FOPT", true);
		test("FOPT", false);
	}

	@Test
	public void testFOT()
	{
		test("FOT", true);
		test("FOT", false);
	}

	@Test
	public void testFOU()
	{
		test("FOU", true);
		test("FOU", false);
	}

	@Test
	public void testFOUT()
	{
		test("FOUT", true);
		test("FOUT", false);
	}

	@Test
	public void testFP()
	{
		test("FP", true);
		test("FP", false);
	}

	@Test
	public void testFPL()
	{
		test("FPL", true);
		test("FPL", false);
	}

	@Test
	public void testFPLT()
	{
		test("FPLT", true);
		test("FPLT", false);
	}

	@Test
	public void testFPT()
	{
		test("FPT", true);
		test("FPT", false);
	}

	@Test
	public void testFT()
	{
		test("FT", true);
		test("FT", false);
	}

	@Test
	public void testFU()
	{
		test("FU", true);
		test("FU", false);
	}

	@Test
	public void testFUT()
	{
		test("FUT", true);
		test("FUT", false);
	}

	@Test
	public void testI()
	{
		test("IFO", true);
		test("IFO", false);
	}

	@Test
	public void testIF()
	{
		test("IF", true);
		test("IF", false);
	}

	@Test
	public void testIFL()
	{
		test("IFL", true);
		test("IFL", false);
	}

	@Test
	public void testIFLT()
	{
		test("IFLT", true);
		test("IFLT", false);
	}

	@Test
	public void testIFO()
	{
		test("IFO", true);
		test("IFO", false);
	}

	@Test
	public void testIFOL()
	{
		test("IFOL", true);
		test("IFOL", false);
	}

	@Test
	public void testIFOLT()
	{
		test("IFOLT", true);
		test("IFOLT", false);
	}

	@Test
	public void testIFOP()
	{
		test("IFOP", true);
		test("IFOP", false);
	}

	@Test
	public void testIFOPL()
	{
		test("IFOPL", true);
		test("IFOPL", false);
	}

	@Test
	public void testIFOPLT()
	{
		test("IFOPLT", true);
		test("IFOPLT", false);
	}

	@Test
	public void testIFOPT()
	{
		test("IFOPT", true);
		test("IFOPT", false);
	}

	@Test
	public void testIFOT()
	{
		test("IFOT", true);
		test("IFOT", false);
	}

	@Test
	public void testIFOU()
	{
		test("IFOU", true);
		test("IFOU", false);
	}

	@Test
	public void testIFOUT()
	{
		test("IFOUT", true);
		test("IFOUT", false);
	}

	@Test
	public void testIFP()
	{
		test("IFP", true);
		test("IFP", false);
	}

	@Test
	public void testIFPL()
	{
		test("IFPL", true);
		test("IFPL", false);
	}

	@Test
	public void testIFPLT()
	{
		test("IFPLT", true);
		test("IFPLT", false);
	}

	@Test
	public void testIFPT()
	{
		test("IFPT", true);
		test("IFPT", false);
	}

	@Test
	public void testIFT()
	{
		test("IFT", true);
		test("IFT", false);
	}

	@Test
	public void testIFU()
	{
		test("IFU", true);
		test("IFU", false);
	}

	@Test
	public void testIFUT()
	{
		test("IFUT", true);
		test("IFUT", false);
	}

	@Test
	public void testIL()
	{
		test("IFOL", true);
		test("IFOL", false);
	}

	@Test
	public void testILT()
	{
		test("IFOLT", true);
		test("IFOLT", false);
	}

	@Test
	public void testIO()
	{
		test("IO", true);
		test("IO", false);
	}

	@Test
	public void testIOL()
	{
		test("IOL", true);
		test("IOL", false);
	}

	@Test
	public void testIOLT()
	{
		test("IOLT", true);
		test("IOLT", false);
	}

	@Test
	public void testIOP()
	{
		test("IOP", true);
		test("IOP", false);
	}

	@Test
	public void testIOPL()
	{
		test("IOPL", true);
		test("IOPL", false);
	}

	@Test
	public void testIOPLT()
	{
		test("IOPLT", true);
		test("IOPLT", false);
	}

	@Test
	public void testIOPT()
	{
		test("IOPT", true);
		test("IOPT", false);
	}

	@Test
	public void testIOT()
	{
		test("IOT", true);
		test("IOT", false);
	}

	@Test
	public void testIOU()
	{
		test("IOU", true);
		test("IOU", false);
	}

	@Test
	public void testIOUT()
	{
		test("IOUT", true);
		test("IOUT", false);
	}

	@Test
	public void testIP()
	{
		test("IFOP", true);
		test("IFOP", false);
	}

	@Test
	public void testIPL()
	{
		test("IFOPL", true);
		test("IFOPL", false);
	}

	@Test
	public void testIPLT()
	{
		test("IFOPLT", true);
		test("IFOPLT", false);
	}

	@Test
	public void testIPT()
	{
		test("IFOPT", true);
		test("IFOPT", false);
	}

	@Test
	public void testIT()
	{
		test("IFOT", true);
		test("IFOT", false);
	}

	@Test
	public void testIU()
	{
		test("IFOU", true);
		test("IFOU", false);
	}

	@Test
	public void testIUT()
	{
		test("IFOUT", true);
		test("IFOUT", false);
	}

	@Test
	public void testL()
	{
		test("L", true);
		test("L", false);
	}

	@Test
	public void testLT()
	{
		test("LT", true);
		test("LT", false);
	}

	@Test
	public void testO()
	{
		test("O", true);
		test("O", false);
	}

	@Test
	public void testOL()
	{
		test("OL", true);
		test("OL", false);
	}

	@Test
	public void testOLT()
	{
		test("OLT", true);
		test("OLT", false);
	}

	@Test
	public void testOP()
	{
		test("OP", true);
		test("OP", false);
	}

	@Test
	public void testOPL()
	{
		test("OPL", true);
		test("OPL", false);
	}

	@Test
	public void testOPLT()
	{
		test("OPLT", true);
		test("OPLT", false);
	}

	@Test
	public void testOPT()
	{
		test("OPT", true);
		test("OPT", false);
	}

	@Test
	public void testOT()
	{
		test("OT", true);
		test("OT", false);
	}

	@Test
	public void testOU()
	{
		test("OU", true);
		test("OU", false);
	}

	@Test
	public void testOUT()
	{
		test("OUT", true);
		test("OUT", false);
	}

	@Test
	public void testP()
	{
		test("P", true);
		test("P", false);
	}

	@Test
	public void testPL()
	{
		test("PL", true);
		test("PL", false);
	}

	@Test
	public void testPLT()
	{
		test("PLT", true);
		test("PLT", false);
	}

	@Test
	public void testPT()
	{
		test("PT", true);
		test("PT", false);
	}

	@Test
	public void testT()
	{
		test("T", true);
		test("T", false);
	}

	@Test
	public void testU()
	{
		test("U", true);
		test("U", false);
	}

	@Test
	public void testUT()
	{
		test("UT", true);
		test("UT", false);
	}

	private void updateWeights(final float[][] w, final float[][] e, final float[] d)
	{
		assertTrue(w.length == e.length);
		assertTrue(w.length == d.length);
		for(int j = 0; j < w.length; j++)
		{
			assertTrue(w[j].length == e[j].length);
			for(int i = 0; i < w[j].length; i++)
				w[j][i] += e[j][i] * d[j];
		}
	}

	private void xbackward(final int t)
	{
		final float[][] ac = a[t];
		final float[][] dc = d[t];
		final float[][] rc = r[t];
		final float[][][] ec = e[t];
		final float[][][] wc = w[t];
		final float[][][] wp = w[t - 1];
		final float[] tgc = tg[t];
		final float[] buf = new float[size];

		// copy previous weights forward;
		copyWeights(w, t - 1, t);

		// out
		assertTrue(tgc.length == ac[out].length);
		assertTrue(tgc.length == dc[out].length);
		for(int j = 0; j < tgc.length; j++)
			dc[out][j] = tgc[j] - ac[out][j];
		// comment out for logistic & xentropy
		// fp(buf, ac[out]);
		// mul(dc[out], dc[out], buf);
		updateWeights(wc[outb], ec[outb], dc[out]);
		updateWeights(wc[outf], ec[outf], dc[out]);

		// mg2
		backpropw(dc[mg2], dc[out], wp[outf]);

		// og2
		if(og)
		{
			backpropm(dc[og2], dc[mg2], ac[mc2]);
			fp(buf, ac[og2]);
			mul(dc[og2], dc[og2], buf);
			updateWeights(wc[og2b], ec[og2b], dc[og2]);
			updateWeights(wc[og2f], ec[og2f], dc[og2]);
			if(ph)
				updateWeights(wc[og2p], ec[og2p], dc[og2]);
			if(gl || ul)
				updateWeights(wc[og2l], ec[og2l], dc[og2]);
		}

		// mc2
		if(og)
			backpropm(dc[mc2], dc[mg2], ac[og2]);
		else
			copy(dc[mc2], dc[mg2]);

		if(ph && !tr)
			backpropw(dc[mc2], dc[og2], wp[og2p]);

		if(ul && !tr)
			backpropw(dc[mc2], dc[og2], wp[og2l]);

		fp(buf, ac[mc2]);
		mul(dc[mc2], dc[mc2], buf);
		copy(dc[st2], dc[mc2]);
		updateWeights(wc[mc2b], ec[mc2b], dc[mc2]);
		updateWeights(wc[mc2f], ec[mc2f], dc[mc2]);

		// fg2
		if(fg)
		{
			copy(dc[fg2], dc[mc2]);
			updateWeights(wc[fg2b], ec[fg2b], dc[fg2]);
			updateWeights(wc[fg2f], ec[fg2f], dc[fg2]);
			updateWeights(wc[fg2p], ec[fg2p], dc[fg2]);
			updateWeights(wc[fg2l], ec[fg2l], dc[fg2]);
		}

		// ig2
		if(ig)
		{
			copy(dc[ig2], dc[mc2]);
			updateWeights(wc[ig2b], ec[ig2b], dc[ig2]);
			updateWeights(wc[ig2f], ec[ig2f], dc[ig2]);
			updateWeights(wc[ig2p], ec[ig2p], dc[ig2]);
			updateWeights(wc[ig2l], ec[ig2l], dc[ig2]);
		}

		// mg1
		backpropwr(dc[mg1], dc[mc2], rc[in2], wp[mc2f]);
		if(!tr)
		{
			if(ig)
				backpropwr(dc[mg1], dc[mc2], rc[ig2], wp[ig2f]);
			if(fg)
				backpropwr(dc[mg1], dc[mc2], rc[fg2], wp[fg2f]);
			if(og)
				backpropw(dc[mg1], dc[og2], wp[og2f]);
		}

		// og1
		if(og)
		{
			backpropm(dc[og1], dc[mg1], ac[mc1]);
			fp(buf, ac[og1]);
			mul(dc[og1], dc[og1], buf);
			updateWeights(wc[og1b], ec[og1b], dc[og1]);
			updateWeights(wc[og1f], ec[og1f], dc[og1]);
			if(ph)
				updateWeights(wc[og1p], ec[og1p], dc[og1]);
			if(gl || ul)
				updateWeights(wc[og1l], ec[og1l], dc[og1]);
		}

		// mc1
		if(og)
			backpropm(dc[mc1], dc[mg1], ac[og1]);
		else
			copy(dc[mc1], dc[mg1]);

		if(ph && !tr)
			backpropw(dc[mc1], dc[og1], wp[og1p]);

		if(ul && !tr)
			backpropw(dc[mc1], dc[og1], wp[og1l]);

		fp(buf, ac[mc1]);
		mul(dc[mc1], dc[mc1], buf);
		copy(dc[st1], dc[mc1]);
		updateWeights(wc[mc1b], ec[mc1b], dc[mc1]);
		updateWeights(wc[mc1f], ec[mc1f], dc[mc1]);

		// fg1
		if(fg)
		{
			copy(dc[fg1], dc[mc1]);
			updateWeights(wc[fg1b], ec[fg1b], dc[fg1]);
			updateWeights(wc[fg1f], ec[fg1f], dc[fg1]);
			updateWeights(wc[fg1p], ec[fg1p], dc[fg1]);
			updateWeights(wc[fg1l], ec[fg1l], dc[fg1]);
		}

		// ig1
		if(ig)
		{
			copy(dc[ig1], dc[mc1]);
			updateWeights(wc[ig1b], ec[ig1b], dc[ig1]);
			updateWeights(wc[ig1f], ec[ig1f], dc[ig1]);
			updateWeights(wc[ig1p], ec[ig1p], dc[ig1]);
			updateWeights(wc[ig1l], ec[ig1l], dc[ig1]);
		}
	}

	private void xforward(final int t)
	{
		final float[][] ac = a[t];
		final float[][] ap = a[t - 1];
		final float[][] rc = r[t];
		final float[][][] ec = e[t];
		final float[][][] ep = e[t - 1];
		final float[][][] wc = w[t - 1];

		// ACTIVATIONS 1

		// ig1
		if(ig)
		{
			addb(ac[ig1], wc[ig1b]);
			if(ph)
				addw(ac[ig1], ap[mc1], wc[ig1p]);
			if(gl)
				addw(ac[ig1], ap[mg1], wc[ig1l]);
			if(ul)
				addw(ac[ig1], ap[mc1], wc[ig1l]);
			addw(ac[ig1], ac[in], wc[ig1f]);
			f(ac[ig1], ac[ig1]);
		}

		// fg1
		if(fg)
		{
			addb(ac[fg1], wc[fg1b]);
			if(ph)
				addw(ac[fg1], ap[mc1], wc[fg1p]);
			if(gl)
				addw(ac[fg1], ap[mg1], wc[fg1l]);
			if(ul)
				addw(ac[fg1], ap[mc1], wc[fg1l]);
			addw(ac[fg1], ac[in], wc[fg1f]);
			f(ac[fg1], ac[fg1]);
		}

		// in1
		addw(ac[in1], ac[in], wc[mc1f]);

		// st1
		if(ig)
			addg(ac[st1], ac[in1], ac[ig1]);
		else
			add(ac[st1], ac[in1]);

		if(fg)
			addg(ac[st1], ap[st1], ac[fg1]);
		else
			add(ac[st1], ap[st1]);

		// mc1
		copy(ac[mc1], ac[st1]);
		addb(ac[mc1], wc[mc1b]);
		f(ac[mc1], ac[mc1]);

		// og1
		if(og)
		{
			addb(ac[og1], wc[og1b]);
			if(ph)
				addw(ac[og1], ac[mc1], wc[og1p]);
			if(gl)
				addw(ac[og1], ap[mg1], wc[og1l]);
			if(ul)
				addw(ac[og1], ac[mc1], wc[og1l]);
			addw(ac[og1], ac[in], wc[og1f]);
			f(ac[og1], ac[og1]);
		}

		// mg1
		if(og)
			addg(ac[mg1], ac[mc1], ac[og1]);
		else
			copy(ac[mg1], ac[mc1]);

		// ELIGIBILITIES 1

		// ig1*
		if(ig)
		{
			// gate previous eligibilities
			if(fg)
			{
				addepg(ec[ig1b], ep[ig1b], ac[fg1]);
				if(ph)
					addepg(ec[ig1p], ep[ig1p], ac[fg1]);
				if(gl || ul)
					addepg(ec[ig1l], ep[ig1l], ac[fg1]);
				addepg(ec[ig1f], ep[ig1f], ac[fg1]);
			}
			else
			{
				addep(ec[ig1b], ep[ig1b]);
				addep(ec[ig1p], ep[ig1p]);
				addep(ec[ig1l], ep[ig1l]);
				addep(ec[ig1f], ep[ig1f]);
			}
			// add new eligibility contributions
			fp(rc[ig1], ac[ig1]);
			mul(rc[ig1], rc[ig1], ac[in1]);
			addebg(ec[ig1b], rc[ig1]);
			if(ph)
				addegd(ec[ig1p], ap[mc1], rc[ig1]);
			if(gl)
				addeg(ec[ig1l], ap[mg1], rc[ig1]);
			if(ul)
				addeg(ec[ig1l], ap[mc1], rc[ig1]);
			addeg(ec[ig1f], ac[in], rc[ig1]);
		}

		// fg1*
		if(fg)
		{
			// gate previous eligibilities
			addepg(ec[fg1b], ep[fg1b], ac[fg1]);
			if(ph)
				addepg(ec[fg1p], ep[fg1p], ac[fg1]);
			if(gl || ul)
				addepg(ec[fg1l], ep[fg1l], ac[fg1]);
			addepg(ec[fg1f], ep[fg1f], ac[fg1]);
			// add new eligibility contributions
			fp(rc[fg1], ac[fg1]);
			mul(rc[fg1], rc[fg1], ap[st1]);
			addebg(ec[fg1b], rc[fg1]);
			if(ph)
				addegd(ec[fg1p], ap[mc1], rc[fg1]);
			if(gl)
				addeg(ec[fg1l], ap[mg1], rc[fg1]);
			if(ul)
				addeg(ec[fg1l], ap[mc1], rc[fg1]);
			addeg(ec[fg1f], ac[in], rc[fg1]);
		}

		// mc1*
		// biases are after copy-point; input weights are not
		addeb(ec[mc1b]);
		// gate previous eligibilities
		if(fg)
			addepg(ec[mc1f], ep[mc1f], ac[fg1]);
		else
			addep(ec[mc1f], ep[mc1f]);
		// add new eligibility contributions
		if(ig)
			copy(rc[in1], ac[ig1]);
		else
			ones(rc[in1]);
		addeg(ec[mc1f], ac[in], rc[in1]);

		// og1*
		// all weights are after copy-point
		if(og)
		{
			addeb(ec[og1b]);
			if(ph)
				added(ec[og1p], ac[mc1]);
			if(gl)
				adde(ec[og1l], ap[mg1]);
			if(ul)
				adde(ec[og1l], ac[mc1]);
			adde(ec[og1f], ac[in]);
		}

		// ACTIVATIONS 2

		// ig2
		if(ig)
		{
			addb(ac[ig2], wc[ig2b]);
			if(ph)
				addw(ac[ig2], ap[mc2], wc[ig2p]);
			if(gl)
				addw(ac[ig2], ap[mg2], wc[ig2l]);
			if(ul)
				addw(ac[ig2], ap[mc2], wc[ig2l]);
			addw(ac[ig2], ac[mg1], wc[ig2f]);
			f(ac[ig2], ac[ig2]);
		}

		// fg2
		if(fg)
		{
			addb(ac[fg2], wc[fg2b]);
			if(ph)
				addw(ac[fg2], ap[mc2], wc[fg2p]);
			if(gl)
				addw(ac[fg2], ap[mg2], wc[fg2l]);
			if(ul)
				addw(ac[fg2], ap[mc2], wc[fg2l]);
			addw(ac[fg2], ac[mg1], wc[fg2f]);
			f(ac[fg2], ac[fg2]);
		}

		// in2
		addw(ac[in2], ac[mg1], wc[mc2f]);

		// st2
		if(ig)
			addg(ac[st2], ac[in2], ac[ig2]);
		else
			add(ac[st2], ac[in2]);

		if(fg)
			addg(ac[st2], ap[st2], ac[fg2]);
		else
			add(ac[st2], ap[st2]);

		// mc2
		copy(ac[mc2], ac[st2]);
		addb(ac[mc2], wc[mc2b]);
		f(ac[mc2], ac[mc2]);

		// og2
		if(og)
		{
			addb(ac[og2], wc[og2b]);
			if(ph)
				addw(ac[og2], ac[mc2], wc[og2p]);
			if(gl)
				addw(ac[og2], ap[mg2], wc[og2l]);
			if(ul)
				addw(ac[og2], ac[mc2], wc[og2l]);
			addw(ac[og2], ac[mg1], wc[og2f]);
			f(ac[og2], ac[og2]);
		}

		// mg2
		if(og)
			addg(ac[mg2], ac[mc2], ac[og2]);
		else
			copy(ac[mg2], ac[mc2]);

		// ELIGIBILITIES 2

		// ig2*
		if(ig)
		{
			// gate previous eligibilities
			if(fg)
			{
				addepg(ec[ig2b], ep[ig2b], ac[fg2]);
				if(ph)
					addepg(ec[ig2p], ep[ig2p], ac[fg2]);
				if(gl || ul)
					addepg(ec[ig2l], ep[ig2l], ac[fg2]);
				addepg(ec[ig2f], ep[ig2f], ac[fg2]);
			}
			else
			{
				addep(ec[ig2b], ep[ig2b]);
				addep(ec[ig2p], ep[ig2p]);
				addep(ec[ig2l], ep[ig2l]);
				addep(ec[ig2f], ep[ig2f]);
			}
			// add new eligibility contributions
			fp(rc[ig2], ac[ig2]);
			mul(rc[ig2], rc[ig2], ac[in2]);
			addebg(ec[ig2b], rc[ig2]);
			if(ph)
				addegd(ec[ig2p], ap[mc2], rc[ig2]);
			if(gl)
				addeg(ec[ig2l], ap[mg2], rc[ig2]);
			if(ul)
				addeg(ec[ig2l], ap[mc2], rc[ig2]);
			addeg(ec[ig2f], ac[mg1], rc[ig2]);
		}

		// fg2*
		if(fg)
		{
			// gate previous eligibilities
			addepg(ec[fg2b], ep[fg2b], ac[fg2]);
			if(ph)
				addepg(ec[fg2p], ep[fg2p], ac[fg2]);
			if(gl || ul)
				addepg(ec[fg2l], ep[fg2l], ac[fg2]);
			addepg(ec[fg2f], ep[fg2f], ac[fg2]);
			// add new eligibility contributions
			fp(rc[fg2], ac[fg2]);
			mul(rc[fg2], rc[fg2], ap[st2]);
			addebg(ec[fg2b], rc[fg2]);
			if(ph)
				addegd(ec[fg2p], ap[mc2], rc[fg2]);
			if(gl)
				addeg(ec[fg2l], ap[mg2], rc[fg2]);
			if(ul)
				addeg(ec[fg2l], ap[mc2], rc[fg2]);
			addeg(ec[fg2f], ac[mg1], rc[fg2]);
		}

		// mc2*
		// biases are after copy-point; input weights are not
		addeb(ec[mc2b]);
		// gate previous eligibilities
		if(fg)
			addepg(ec[mc2f], ep[mc2f], ac[fg2]);
		else
			addep(ec[mc2f], ep[mc2f]);
		// add new eligibility contributions
		if(ig)
			copy(rc[in2], ac[ig2]);
		else
			ones(rc[in2]);
		addeg(ec[mc2f], ac[mg1], rc[in2]);

		// og2*
		// all weights are after copy-point
		if(og)
		{
			addeb(ec[og2b]);
			if(ph)
				added(ec[og2p], ac[mc2]);
			if(gl)
				adde(ec[og2l], ap[mg2]);
			if(ul)
				adde(ec[og2l], ac[mc2]);
			adde(ec[og2f], ac[mg1]);
		}

		// ACTIVATIONS OUT

		// out
		addb(ac[out], wc[outb]);
		addw(ac[out], ac[mg2], wc[outf]);

		// ELIGIBILITIES OUT

		// out*

		addeb(ec[outb]);
		adde(ec[outf], ac[mg2]);
	}
}
