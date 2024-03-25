package RN.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import RN.nodes.PixelNode;

/**
 * @author Eric Marchand
 *
 */
public class MatrixUtils {
	
	public static <T> List<T> toList(T[][] matrix) {
		
	    List<T> list = new ArrayList<T>();
	    for (T[] array : matrix) {
	        list.addAll(Arrays.asList(array));
	    }
	    return list;
	}
	
	public static <T> List<T> toList(T[][] matrix, int sampling) {
		List<T> list = new ArrayList<T>();
		List<T> row = null;
	    for (T[] array : matrix) {
	    	row = Arrays.asList(array).stream().filter(node -> ((PixelNode) node).getX() % sampling == 0 ).filter(node -> ((PixelNode) node).getY() % sampling == 0).collect(Collectors.toList());
	        if(!row.isEmpty())
	        	list.addAll(row);
	    }
	    return list;
		
	}
	

}
