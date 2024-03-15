package dmonner.xlbp;

import dmonner.xlbp.layer.InputLayer;
import dmonner.xlbp.util.MatrixTools;

public class Input
{
	private final InputLayer layer;
	private final float[] value;

	public Input(final InputLayer layer, final float[] value)
	{
		this.layer = layer;
		this.value = value;

		if(value.length != layer.size())
			throw new IllegalArgumentException("Incorrect Input Size; expected " + layer.size() + " for "
					+ layer.getName() + ", got " + value.length);
	}

	public void apply()
	{
		layer.setInput(value);
	}

	@Override
	public boolean equals(final Object other)
	{
		if(other instanceof Input)
		{
			final Input that = (Input) other;
			return that.layer == this.layer;
		}

		return false;
	}

	public InputLayer getLayer()
	{
		return layer;
	}

	public float[] getValue()
	{
		return value;
	}

	@Override
	public int hashCode()
	{
		return layer.hashCode();
	}

	@Override
	public String toString()
	{
		return layer.getName() + ": " + MatrixTools.toString(value);
	}
}
