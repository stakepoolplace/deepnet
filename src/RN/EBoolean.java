package RN;

public enum EBoolean {
	
	YES(Boolean.TRUE),
	NO(Boolean.FALSE);
	
	private Boolean value = null;
	
	EBoolean(boolean value){
		this.value = value;
	}

	public Boolean getValue() {
		return value;
	}

	public void setValue(Boolean value) {
		this.value = value;
	}
	
	

}
