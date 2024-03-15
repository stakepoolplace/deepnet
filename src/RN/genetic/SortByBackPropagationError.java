package RN.genetic;

import java.util.Comparator;

import RN.INetwork;

public class SortByBackPropagationError implements Comparator<INetwork> {

	@Override
	public int compare(INetwork o1, INetwork o2) {
		
		if(o1.getAbsoluteError() - o2.getAbsoluteError() > 0)
			return 1;
		else if( o1.getAbsoluteError() - o2.getAbsoluteError() == 0)
			return 0;
		else
			return -1;
	}

}
