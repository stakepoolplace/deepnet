package RN.linkage;

/**
 * @author Eric Marchand
 *
 */
public enum ELinkage {
	
	ONE_TO_ONE("RN.linkage.OneToOneLinkage"),
	ONE_TO_ONE_FILTER("RN.linkage.OneToOneFilterLinkage"),
	//ONE_TO_ONE_FETCH_AREA("RN.linkage.OneToOneFetchAreaLinkage"),
	ONE_TO_ONE_FETCH_OCTAVE_AREA("RN.linkage.OneToOneFetchOctaveAreaLinkage"),
	ONE_TO_ONE_OCTAVE("RN.linkage.OneToOneOctaveAreaLinkage"),
	MANY_TO_MANY("RN.linkage.FullFanOutLinkage"),
	MANY_TO_MANY_INTRA_AREA("RN.linkage.FullFanOutLinkageIntraArea"),
	MANY_TO_MANY_OCTAVE("RN.linkage.FullFanOutOctaveAreaLinkage"),
	FIRST_DERIVATED_GAUSSIAN("RN.linkage.FirstDerivatedGaussianLinkage"),
	SIMPLE_CONTOUR("RN.linkage.ContourLinkage"), 
	DOG("RN.linkage.DifferenceOfGaussianLinkage"),
	DOG_STATIC("RN.linkage.DOGStaticLinkage"),
	SONAG("RN.linkage.SumOfNegativeAndGaussianLinkage"),
	LOG("RN.linkage.LaplacianOfGaussianLinkage"),
	LOG_STATIC("RN.linkage.LOGStaticLinkage"),
	GAUSSIAN("RN.linkage.GaussianLinkage"),
	HESSIAN_COURBURE("RN.linkage.HessianCourbureLinkage"),
	GENERIC("RN.linkage.GenericFilterLinkage"),
	MAX_POOLING("RN.linkage.MaxPoolingLinkage"),
	MAX("RN.linkage.MaxLinkage"),
	MAP("RN.linkage.MapLinkage"),
	SAMPLING("RN.linkage.SamplingLinkage"),
	CONVOLUTION("RN.linkage.ConvolutionLinkage"),
	SUBSAMPLING("RN.linkage.SubsamplingLinkage"),
	
	// Vision
	BIPOLAR("RN.linkage.vision.BiPolarCellLinkage"),
	GANGLIONARY("RN.linkage.vision.GanglionaryCellLinkage"), 
	V1_ORIENTATIONS("RN.linkage.vision.V1OrientationsCellLinkage"),
	LOG_GABOR("RN.linkage.GaborLogLinkage"),
	LOG_GABOR2("RN.linkage.GaborLogLinkage2"),
	LOG_POLAR("RN.linkage.LogPolarLinkage"),
	CARTESIAN_TO_POLAR("RN.linkage.CartesianToPolarImageLinkage"),
	CARTESIAN_TO_LOG_POLAR("RN.linkage.CartesianToLogPolarImageLinkage")
	;
	
	String classPath = null;
	
	ELinkage(String classPath){
		this.classPath = classPath;
	}

	public String getClassPath() {
		return classPath;
	}

	public void setClassPath(String classPath) {
		this.classPath = classPath;
	}


}
