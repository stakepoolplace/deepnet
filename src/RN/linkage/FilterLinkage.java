package RN.linkage;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import RN.IAreaSquare;
import RN.ILayer;
import RN.Identification;
import RN.dataset.inputsamples.ESamples;
import RN.nodes.INode;
import RN.nodes.IPixelNode;
import RN.nodes.PixelNode;

public abstract class FilterLinkage extends Linkage implements IFilterLinkage{
	
	public static class FilterIndex {
		private Identification id = null;
		private Integer idFilter = null;
		public FilterIndex(Identification id, Integer idFilter){
			this.id = id;
			this.idFilter = idFilter;
		}
		public Identification getId() {
			return id;
		}
		public void setId(Identification id) {
			this.id = id;
		}
		public Integer getIdFilter() {
			return idFilter;
		}
		public void setIdFilter(Integer idFilter) {
			this.idFilter = idFilter;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((id == null) ? 0 : id.hashCode());
			result = prime * result + ((idFilter == null) ? 0 : idFilter.hashCode());
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			FilterIndex other = (FilterIndex) obj;
			if (id == null) {
				if (other.id != null)
					return false;
			} else if (!id.equals(other.id))
				return false;
			if (idFilter == null) {
				if (other.idFilter != null)
					return false;
			} else if (!idFilter.equals(other.idFilter))
				return false;
			return true;
		}
	}
	
	// Tableau de filtres, matrices de dimension impaire.
	private final static Map<FilterIndex, Filter> filters = new HashMap<FilterIndex, Filter>();

	public final static int ID_FILTER_V1Orientation = 0;
	public final static int ID_FILTER_DOG_0 = 1;
	public final static int ID_FILTER_DOG_1 = 2;
	public final static int ID_FILTER_DOG_STATIC = 3;
	public final static int ID_FILTER_FIRST_DERIVATED_GAUSSIAN = 4;
	public final static int ID_FILTER_GAUSSIAN = 5;
	public final static int ID_FILTER_LOG = 6;
	public final static int ID_FILTER_LOG_STATIC = 7;
	public final static int ID_FILTER_GENERIC = 8;
	public final static int ID_FILTER_Gxx = 9;
	public final static int ID_FILTER_Gyy = 10;
	public final static int ID_FILTER_Gxy = 11;
	public final static int ID_FILTER_SAMPLING_2 = 12;
	public final static int ID_FILTER_SONAG = 13;
	public final static int ID_FILTER_MAX_POOLING = 14;
	public final static int ID_FILTER_GABOR_LOG = 15;
	public final static int ID_FILTER_CONVOLUTION = 16;
	
	protected ESamples eSampleFunction = null;
	
	@Override
	public abstract void sublayerFanOutLinkage(INode thisNode, ILayer sublayer) ;
	
	@Override
	public abstract void sublayerFanOutLinkage(INode thisNode, ILayer sublayer, long initFireTimeT);

	@Override
	public abstract void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer);

	@Override
	public abstract void nextLayerFanOutLinkage(INode thisNode, ILayer nextlayer, long initFireTimeT);
	
	
	public Double processFilter(ESamples filterFunction, IAreaSquare subArea, IPixelNode sublayerNode){
		throw new RuntimeException("Filtre absent");
	}
	
	public Double processFilter(ESamples filterFunction, IAreaSquare subArea, IPixelNode sublayerNode, Double... params){
		throw new RuntimeException("Filtre absent");
	}

	public void setESampleFunction(ESamples eSampleFunction){
		this.eSampleFunction = eSampleFunction;
	}
	
	public Filter getFilter(FilterIndex idFilter) {
		return filters.get(idFilter);
	}
	
	public static Map<FilterIndex, Filter> getFilters() {
		return filters;
	}
	

	public void setFilter(FilterIndex idx, Filter filter) {
		this.filters.put(idx, filter);
	}
	
	public static void removeFilter(FilterIndex idx){
		filters.remove(idx);
	}
	
	public void initFilter(IFilterLinkage linkage, int idFilter, ESamples filterFunction, IPixelNode thisNode, IAreaSquare subArea, Double... params) {
		initFilter(linkage, idFilter, filterFunction, 0.001D, thisNode, subArea, params);
	}
	
	
	public void initFilter(IFilterLinkage linkage, int idFilter, ESamples filterFunction, Double cacheSensibility, IPixelNode thisNode, IAreaSquare subArea, Double... params) {
		
		
		FilterIndex idx = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		
		if (this.getFilter(idx) == null) {
			
			Filter existingFilter = Filter.returnExistingFilter(filterFunction, params);
			Filter filter = null;
			if(existingFilter == null){
				
				filter = new Filter(idFilter, filterFunction, new Double[subArea.getWidthPx()][subArea.getHeightPx()], params);
				
				double filterValue = 0D;
				
				List<INode> nodeList = subArea.getNodes();
				INode sublayerNode = null;
				for (int index = 0; index < nodeList.size(); index++) {
					
					sublayerNode = nodeList.get(index);
					
					filterValue = linkage.processFilter(filterFunction, (IPixelNode) sublayerNode, params);

					// Ajout des valeurs discretes du filtre dans le cache
					if (Math.abs(filterValue) >= cacheSensibility) {
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), filterValue);
					}else{
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), 0D);
					}
				}
				
				filter.resizeFilter();
				
				this.setFilter(idx,  filter);
				
			}else{
				
				this.setFilter(idx,  existingFilter);
				
			}

			this.getFilter(idx).filterToString();
		}
		
	}
	
	public void initFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, int width, int height, float[] values, Double... params) {
		
		
		FilterIndex idx = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		
		if (this.getFilter(idx) == null) {
			
			Filter existingFilter = Filter.returnExistingFilter(ESamples.NONE, params);
			Filter filter = null;
			if(existingFilter == null){
				
				filter = new Filter(idFilter, ESamples.NONE, new Double[width][height], params);
				
				for(int i=0; i < width; i++){
					for(int j=0; j < height; j++){
						filter.setValue(i, j, (double) values[i * j + j]);
					}
				}
				
				
				filter.resizeFilter();
				
				this.setFilter(idx,  filter);
				
			}else{
				
				this.setFilter(idx,  existingFilter);
				
			}

			this.getFilter(idx).filterToString();
		}
		
	}
	
	public void initFilter(IFilterLinkage linkage, int idFilter, IPixelNode thisNode, int width, int height, Double[][] values, Double... params) {
		
		
		FilterIndex idx = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		
		if (this.getFilter(idx) == null) {
			
			Filter existingFilter = Filter.returnExistingFilter(ESamples.NONE, params);
			Filter filter = null;
			if(existingFilter == null){
				
				filter = new Filter(idFilter, ESamples.NONE, new Double[width][height], params);
				
				for(int i=0; i < values.length; i++){
					for(int j=0; j < values[i].length; j++){
						filter.setValue(i, j, values[i][j]);
					}
				}
				
				
				filter.resizeFilter();
				
				this.setFilter(idx,  filter);
				
			}else{
				
				this.setFilter(idx,  existingFilter);
				
			}

			this.getFilter(idx).filterToString();
		}
		
	}
	
	public void initCompositeFilter(IFilterLinkage linkage, int idFilter, ESamples op, ESamples filterFunction, ESamples filter2Function, IPixelNode thisNode, List<IPixelNode> subNodes, Double... params) {
		
		FilterIndex idx = new FilterIndex(thisNode.getAreaSquare().getIdentification(), idFilter);
		
		if (this.getFilter(idx) == null) {
			
			Double[] params0 = Arrays.copyOfRange(params, 0, (params.length / 2));
			Double[] params1 = Arrays.copyOfRange(params, (params.length / 2), params.length);
			
			Filter existingFilter = Filter.returnExistingFilter(filterFunction, params0);
			Filter existingFilter2 = Filter.returnExistingFilter(filter2Function, params1);
			Filter filter = null;
			if(existingFilter == null || existingFilter2 == null){
				
				IAreaSquare area = subNodes.get(0).getAreaSquare();
				
				filter = new Filter(idFilter, filterFunction, new Double[area.getWidthPx()][area.getHeightPx()], params);
				
				double filterValue = 0D;
				double filterValue0 = 0D;
				double filterValue1 = 0D;
				
				
				IPixelNode sublayerNode = null;
				for (int index = 0; index < subNodes.size(); index++) {
					
					sublayerNode = subNodes.get(index);

					if(existingFilter == null){
						filterValue0 = linkage.processFilter(filterFunction, sublayerNode, params0);
					}else{
						filterValue0 = existingFilter.getValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY());
					}
					
					if(existingFilter2 == null){
						filterValue1 = linkage.processFilter(filter2Function, sublayerNode, params1);
					}else{
						filterValue1 = existingFilter2.getValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY());
					}
					
					
					if(op == ESamples.SUBSTRACT)
						filterValue =  filterValue0 - filterValue1;
					else if(op == ESamples.ADD)
						filterValue = filterValue0 + filterValue1;

					// Ajout des valeurs discretes du filtre dans le cache
					if (Math.abs(filterValue) >= 0.0001D) {
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), filterValue);
					}else{
						filter.setValue(((PixelNode) sublayerNode).getX(), ((PixelNode) sublayerNode).getY(), 0D);
					}
				}
				
				
				this.setFilter(idx,  filter);
				
			}else{
				this.setFilter(idx,  existingFilter);
			}
			
			
			//filterToString(idFilter);
		}
		
	}
	
	
	public Double getFilterValue(FilterIndex index, EFilterPosition filterPosition, IPixelNode thisNode, IPixelNode sourceNode) {
		
		
		int filterWidth = this.getFilter(index).getWidth();
		int filterHeight = this.getFilter(index).getHeight();
		
		int sourceX;
		int sourceY;
		if(sampling != null && sampling != 1){
			sourceX = ((PixelNode) sourceNode).getX() / sampling;
			sourceY = ((PixelNode) sourceNode).getY() / sampling;
		}else{
			sourceX = ((PixelNode) sourceNode).getX();
			sourceY = ((PixelNode) sourceNode).getY();
		}
		
		int targetX = ((PixelNode) thisNode).getX();
		int targetY = ((PixelNode) thisNode).getY();
		
		
		int valueX = -1;
		int valueY = -1;
		if(filterPosition == EFilterPosition.CENTER){
		 
			valueX =  targetX + ((filterWidth - 1) / 2) - sourceX;
			valueY =  targetY + ((filterHeight - 1) / 2) - sourceY;
		
		}else if(filterPosition == EFilterPosition.TOP_LEFT){
			
			 valueX =  sourceX - targetX;
			 valueY =  sourceY - targetY;
		}
		
		
		if(valueX >= 0D && valueX < filterWidth && valueY >= 0D && valueY < filterHeight){
			return this.getFilter(index).getValue(valueX, valueY);
		}
		
		return 0D;
			
	}
	
	

	
}
