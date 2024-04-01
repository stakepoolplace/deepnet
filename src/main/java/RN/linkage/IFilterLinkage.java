package RN.linkage;

import RN.IAreaSquare;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.FilterLinkage.FilterIndex;
import RN.nodes.IPixelNode;

/**
 * @author Eric Marchand
 *
 */
public interface IFilterLinkage {
	
	Double processFilter(ESamples filterFunction, IPixelNode sublayerNode, Double... params);
	
	Double processFilter(ESamples filterFunction, IAreaSquare subArea, IPixelNode sublayerNode);
	
	Double processFilter(ESamples filterFunction, IAreaSquare subArea, IPixelNode sublayerNode, Double... params);
	
	void setESampleFunction(ESamples eSampleFunction);
	
	void setSampling(Integer sampling);
	
	Integer getSampling();
	
	Double getFilterValue(FilterIndex index, EFilterPosition filterPosition, IPixelNode thisNode, IPixelNode sourceNode);
	
	Boolean isWeightModifiable();
	
	Filter getFilter(FilterIndex idFilter);
	
	void initFilter(IFilterLinkage linkage, int idFilter, ESamples filterFunction, IPixelNode thisNode, IAreaSquare subArea, Double... params);
	
	void initFilter(IFilterLinkage linkage, int idFilter, ESamples filterFunction, Double cacheSensibility, IPixelNode thisNode, IAreaSquare subArea, Double... params);
	
	void initFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, int width, int height, Double[][] values, Double... params);
	
	void initFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, int width, int height, float[] values, Double... params);
	
}
