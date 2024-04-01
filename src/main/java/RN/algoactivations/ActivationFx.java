package RN.algoactivations;


/**
 * @author Eric Marchand
 *
 */
public class ActivationFx {

	private ActivationFx instance;
	private String name;
	private EActivation fx;
	
	public ActivationFx(String name, EActivation fx){
		this.name = name;
		this.fx = fx;
		this.instance = this;
	}

	@Override
	public String toString(){
		return this.name;
	}
	
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public EActivation getFx() {
		return fx;
	}

	public void setFx(EActivation fx) {
		this.fx = fx;
	}
	
	public ActivationFx getByEnum(EActivation fx){
		if(instance.getFx() == fx)
			return instance;
		return null;
	}
	
	
}
