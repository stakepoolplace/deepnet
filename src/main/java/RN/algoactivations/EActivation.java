package RN.algoactivations;

import RN.IArea;
import RN.links.Link;

/**
 * @author Eric Marchand
 *
 */
public enum EActivation {

	
	COS, SIN, IDENTITY, EXP, SOFTMAX, HEAVISIDE, SYGMOID_0_1, SYGMOID_1_1, TANH, LINEAR, SYGMOID_0_1_INVERSE, SYGMOID_0_1_NEGATIVE, NEGATIVE, RLU, LEAKY_RELU, SOFTPLUS, ELU;
	
	public static EActivation getEnum(String function){
	
		if (function.equalsIgnoreCase(EActivation.HEAVISIDE.name()))
		return EActivation.HEAVISIDE;
	else if (function.equalsIgnoreCase(EActivation.SYGMOID_0_1.name()))
		return EActivation.SYGMOID_0_1;
	else if (function.equalsIgnoreCase(EActivation.SYGMOID_0_1_INVERSE.name()))
		return EActivation.SYGMOID_0_1_INVERSE;
	else if (function.equalsIgnoreCase(EActivation.SYGMOID_1_1.name()))
		return EActivation.SYGMOID_1_1;
	else if (function.equalsIgnoreCase(EActivation.TANH.name()))
		return EActivation.TANH;
	else if (function.equalsIgnoreCase(EActivation.IDENTITY.name()))
		return EActivation.IDENTITY;
	else if (function.equalsIgnoreCase(EActivation.EXP.name()))
		return EActivation.EXP;
	else if (function.equalsIgnoreCase(EActivation.SIN.name()))
		return EActivation.SIN;
	else if (function.equalsIgnoreCase(EActivation.COS.name()))
		return EActivation.COS;
	else if (function.equalsIgnoreCase(EActivation.LINEAR.name()))
		return EActivation.LINEAR;		
	else if (function.equalsIgnoreCase(EActivation.NEGATIVE.name()))
		return EActivation.NEGATIVE;
	else if (function.equalsIgnoreCase(EActivation.SYGMOID_0_1_NEGATIVE.name()))
		return EActivation.SYGMOID_0_1_NEGATIVE;
	else if (function.equalsIgnoreCase(EActivation.RLU.name()))
		return EActivation.RLU;
	else if (function.equalsIgnoreCase(EActivation.LEAKY_RELU.name()))
		return EActivation.LEAKY_RELU;		
	else if (function.equalsIgnoreCase(EActivation.SOFTMAX.name()))
		return EActivation.SOFTMAX;		
	else if (function.equalsIgnoreCase(EActivation.SOFTPLUS.name()))
		return EActivation.SOFTPLUS;	
	else if (function.equalsIgnoreCase(EActivation.ELU.name()))
		return EActivation.ELU;
		
		return null;
	
	}
	

	public static IActivation getPerformer(EActivation function) {
		
		IActivation performer = null;
		
		if (function == EActivation.HEAVISIDE)
			performer = new HeaviSidePerformer();
		else if (function == EActivation.SYGMOID_0_1)
			performer = new SygmoidPerformer();
		else if (function == EActivation.SYGMOID_0_1_INVERSE)
			performer = new SygmoidInversePerformer();
		else if (function == EActivation.SYGMOID_1_1)
			performer = new SygmoidPerformer2();
		else if (function == EActivation.TANH)
			performer = new TanHPerformer();
		else if (function == EActivation.IDENTITY)
			performer = new IdentityPerformer();
		else if (function == EActivation.EXP)
			performer = new ExponentialPerformer();
		else if (function == EActivation.SIN)
			performer = new SinusPerformer();
		else if (function == EActivation.COS)
			performer = new CosinusPerformer();	
		else if (function == EActivation.LINEAR)
			performer = new LinearPerformer();
		else if (function == EActivation.NEGATIVE)
			performer = new NegativePerformer();
		else if (function == EActivation.SYGMOID_0_1_NEGATIVE)
			performer = new SygmoidNegativePerformer();
		else if (function == EActivation.RLU)
			performer = new ReLUPerformer();
		else if (function == EActivation.LEAKY_RELU)
			performer = new LeakyReLUPerformer();	
		else if (function == EActivation.SOFTPLUS)
			performer = new SoftPlusPerformer();		
		else if (function == EActivation.ELU)
			performer = new ELUPerformer();
		
		return performer;
	}
	
	public static IActivation getAreaPerformer(EActivation function, IArea area) {
		
		IActivation performer = null;
		
	
		if (function == EActivation.SOFTMAX)
			performer = new SoftMaxPerformer(area);
		
		
		return performer;
	}
	
}
