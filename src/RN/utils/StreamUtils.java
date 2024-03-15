package RN.utils;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

public class StreamUtils {

	public StreamUtils() {
	}

//	use it like this:
//
//		//for lists
//		List<String> stringList = Arrays.asList("1","2","3");
//		List<Integer> integerList = convertList(stringList, s -> Integer.parseInt(s));
//
//		//for arrays
//		String[] stringArr = {"1","2","3"};
//		Double[] doubleArr = convertArray(stringArr, Double::parseDouble, Double[]::new);
//		Note that  s -> Integer.parseInt(s) could be replace with Integer::parseInt (see Method references)	
	
	// for lists
	public static <T, U> List<U> convertList(List<T> from, Function<T, U> func) {
		return from.stream().map(func).collect(Collectors.toList());
	}

	// for arrays
	public static <T, U> U[] convertArray(T[] from, Function<T, U> func, IntFunction<U[]> generator) {
		return Arrays.stream(from).map(func).toArray(generator);
	}

}
