package RN.linkage;

public enum ELinkageBetweenAreas {
	
	// DEFAULT MAPPING
	// layer[m].area[n0] <---> layer[m+1].area[n]  : with n0 a constant = {LINKAGETARGETEDAREA else 0}
	ONE_TO_MANY, 
	
	
	// layer[m].area[n] <---> layer[m+1].area[n]
	ONE_TO_ONE,
	
	
	// layer[m].area[n] <---> layer[m+1].area[n0] : with n0 a constant = {LINKAGETARGETEDAREA else 0}
	MANY_TO_ONE;

}
