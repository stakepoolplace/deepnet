package dmonner.xlbp;

import java.util.Random;

public class UniformWeightInitializer implements WeightInitializer
{
	public static final long serialVersionUID = 1L;

	public final Random rand;
	public final float p;
	public final float min;
	public final float max;

	public UniformWeightInitializer()
	{
		this(new Random(), 1F);
	}

	public UniformWeightInitializer(final float p)
	{
		this(new Random(), p, -0.1F, +0.1F);
	}

	public UniformWeightInitializer(final float p, final float min, final float max)
	{
		this(new Random(), p, min, max);
	}

	public UniformWeightInitializer(final Random rand)
	{
		this(rand, 1F);
	}

	public UniformWeightInitializer(final Random rand, final float p)
	{
		this(rand, p, -0.1F, +0.1F);
	}

	public UniformWeightInitializer(final Random rand, final float p, final float min, final float max)
	{
		if(p < 0F || p > 1F)
			throw new IllegalArgumentException("p must be in [0, 1].");

		this.rand = rand;
		this.p = p;
		this.min = min;
		this.max = max;
	}

	@Override
	public boolean fullConnectivity()
	{
		return p == 1F;
	}

	@Override
	public boolean newWeight(final int j, final int i)
	{
		return rand.nextFloat() < p;
	}

	@Override
	public float randomWeight(final int j, final int i)
	{
		return rand.nextFloat() * (max - min) + min;
	}
}
