package dmonner.xlbp.layer;

import dmonner.xlbp.NetworkCopier;
import dmonner.xlbp.util.NoiseGenerator;

public class NoisyInputLayer extends InputLayer
{
	private static final long serialVersionUID = 1L;

	private final NoiseGenerator noise;

	public NoisyInputLayer(final NoisyInputLayer that, final NetworkCopier copier)
	{
		super(that, copier);
		this.noise = that.noise;
	}

	public NoisyInputLayer(final String name, final int size, final NoiseGenerator noise)
	{
		super(name, size);
		this.noise = noise;
	}

	@Override
	public NoisyInputLayer copy(final NetworkCopier copier)
	{
		return new NoisyInputLayer(this, copier);
	}

	@Override
	public NoisyInputLayer copy(final String nameSuffix)
	{
		return copy(new NetworkCopier(nameSuffix));
	}

	@Override
	public void setInput(final float[] activations)
	{
		for(int i = 0; i < size(); i++)
			y[i] = activations[i] + noise.next();
		super.setFilled();
	}
}
