package RN.linkage.vision;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * @author Eric Marchand
 *
 */
public class Histogram {
	
	
	private Map<Double, Double> histogram = new HashMap<Double,Double>();
	
	private Double maxOrientationValue = null;
	private Double maxOrientationKey = null;

	public int size() {
		return histogram.size();
	}

	public boolean isEmpty() {
		return histogram.isEmpty();
	}

	public boolean containsKey(Object key) {
		return histogram.containsKey(key);
	}

	public boolean containsValue(Object value) {
		return histogram.containsValue(value);
	}

	public Double get(Object key) {
		return histogram.get(key);
	}

	public Double put(Double key, Double value) {
		
		if(maxOrientationValue == null || value > maxOrientationValue){
			maxOrientationKey = key;
			maxOrientationValue = value;
		}
		
		return histogram.put(key, value);
	}

	public Double remove(Object key) {
		
		if(key == maxOrientationKey){
			maxOrientationKey = null;
			maxOrientationValue = null;
		}
		
		return histogram.remove(key);
	}

	private void putAll(Map<? extends Double, ? extends Double> m) {
		histogram.putAll(m);
	}

	public void clear() {
		histogram.clear();
	}

	public Set<Double> keySet() {
		return histogram.keySet();
	}

	public Collection<Double> values() {
		return histogram.values();
	}

	public Set<Entry<Double, Double>> entrySet() {
		return histogram.entrySet();
	}

	public boolean equals(Object o) {
		return histogram.equals(o);
	}

	public int hashCode() {
		return histogram.hashCode();
	}

	public Double putIfAbsent(Double key, Double value) {
		
		if(maxOrientationValue == null || value > maxOrientationValue){
			maxOrientationKey = key;
			maxOrientationValue = value;
		}
		
		return histogram.putIfAbsent(key, value);
	}
	
	public Double getMaxOrientationKey(){
		
		return maxOrientationKey;
	}
	
	public Double getMaxOrientationValue(){
		
		if(maxOrientationKey == null)
			return null;
		
		return histogram.get(maxOrientationKey);
	}

	@Override
	public String toString() {
		return String.format("Histogram [histogram=%s]", histogram);
	}



}
