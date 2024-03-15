package dmonner.xlbp;

public interface InputComponent extends Component
{
	@Override
	public InputComponent copy(String nameSuffix);

	@Override
	public InputComponent copy(NetworkCopier copier);

	public void setInput(final float[] activations);
}
