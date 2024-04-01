package RN.linkage.vision;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Eric Marchand
 *
 */
public class KeyPointDescriptor {
	
	List<Histogram> histograms = null;

	public KeyPointDescriptor() {
		this.histograms = new ArrayList<Histogram>();
	}

	public void concatenateHistogram(Histogram histogram) {
		histograms.add(histogram);
	}

	@Override
	public String toString() {
		return "KeyPointDescriptor [descriptor=" + histograms + "]";
	}

	public List<Histogram> getHistograms() {
		return histograms;
	}

	public void setHistograms(List<Histogram> histograms) {
		this.histograms = histograms;
	}



}
