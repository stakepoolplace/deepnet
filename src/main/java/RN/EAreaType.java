package RN;

/**
 * @author Eric Marchand
 * 
 */
public enum EAreaType {
	
	INLINE("RN.Area"), SQUARE("RN.AreaSquare");
	
	String classPath = null;
	
	EAreaType(String classPath){
		this.classPath = classPath;
	}

	public String getClassPath() {
		return classPath;
	}

	public void setClassPath(String classPath) {
		this.classPath = classPath;
	}

}
