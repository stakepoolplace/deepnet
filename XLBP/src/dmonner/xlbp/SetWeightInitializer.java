package dmonner.xlbp;

public class SetWeightInitializer implements WeightInitializer
{
	private static final long serialVersionUID = 1L;

	private final float[][] w;
	private final boolean full;

	public SetWeightInitializer(final float[][] w)
	{
		this.w = w;
		this.full = checkFull();
	}

	public SetWeightInitializer(final float[][] w, final boolean showFull)
	{
		this.w = w;
		this.full = showFull;
	}

	private boolean checkFull()
	{
		for(int j = 0; j < w.length; j++)
			for(int i = 0; i < w[j].length; i++)
				if(!newWeight(j, i))
					return false;

		return true;
	}

	@Override
	public boolean fullConnectivity()
	{
		return full;
	}

	@Override
	public boolean newWeight(final int j, final int i)
	{
		final float v = w[j][i];
		return !Float.isNaN(v) && !Float.isInfinite(v) && v != 0F;
	}

	@Override
	public float randomWeight(final int j, final int i)
	{
		return w[j][i];
	}

}
