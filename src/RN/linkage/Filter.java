package RN.linkage;

import java.util.Arrays;
import java.util.Map.Entry;

import RN.NetworkElement;
import RN.algoactivations.EActivation;
import RN.dataset.inputsamples.ESamples;
import RN.linkage.FilterLinkage.FilterIndex;
import RN.nodes.ImageNode;

/**
 * @author Eric Marchand
 *
 */
public class Filter extends NetworkElement{
	
	private Integer id = null;
	
	private ESamples filterFunction = null;
	
	private Double[][] values = null;
	
	private Double[] parameters = null;
	
	private Integer width = null;
	
	private Integer height = null;
	
	private int maxX = 0;
	
	private int minX = 0;
	
	private int maxY = 0;
	
	private int minY = 0;
	
	public Filter(int id, ESamples filterFunction, Double[][] values, Double... parameters){
		this.id = id;
		this.filterFunction = filterFunction;
		this.values = values;
		this.parameters = parameters;
		this.minX = values.length;
		this.minY = values[0].length;
		this.width = values.length;
		this.height = values[0].length;
	}
	
	public Filter(int id, Double[][] values, Double... parameters){
		this.id = id;
		this.values = values;
		this.parameters = parameters;
		this.minX = values.length;
		this.minY = values[0].length;
		this.width = values.length;
		this.height = values[0].length;
	}

	public int getLength(){
		return this.values.length;
	}
	
	public Double getValue(int x, int y){
		return this.values[x][y];
	}
	
	public void filterToImage(Integer scale){
		
		ImageNode img = new ImageNode(EActivation.IDENTITY, width, height);
		
		img.getStage().setTitle("Filtre #"+ id +" : "+ height + " x " + width + "  params="+ Arrays.deepToString(parameters));
		
		if(scale != null && scale > 1)
			img.scaleImage(scale);
		
		img.insertDataFilter(this);
		img.drawImageData(null);
	}
	
	
	public void filterToString(){
		
		
		System.out.println("Filtre #"+ id +" : "+ height + " x " + width + "  params="+ Arrays.deepToString(parameters));
		
		for (int idy = 0; idy < height; idy++) {
			for (int idx = 0; idx < width; idx++) {
				Double value = values[idx][idy];
				if(value == null || value == 0D)
					System.out.print(" . ");
				else if(value < 0D)
					System.out.printf(" %.2f ", value);
				else
					System.out.printf(" %.2f ", value);
				
				System.out.print("\t");
			}
			System.out.print("\n");
		}
			
	}



	public Integer getId() {
		return id;
	}



	public void setId(Integer id) {
		this.id = id;
	}



	public ESamples getFilterFunction() {
		return filterFunction;
	}



	public void setFilterFunction(ESamples filterFunction) {
		this.filterFunction = filterFunction;
	}



	public Double[][] getValues() {
		return values;
	}



	public void setValues(Double[][] values) {
		this.values = values;
	}
	
	public void setValue(int x, int y, Double value) {
		
		if(value != 0D){
			
			if(x < minX)
				minX = x;
			
			if(x > maxX)
				maxX = x;
			
			if(y < minY)
				minY = y;
			
			if(y > maxY)
				maxY = y;
		}
		
		this.values[x][y] = value;
	}



	public Double[] getParameters() {
		return parameters;
	}



	public void setParameters(Double[] parameters) {
		this.parameters = parameters;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((filterFunction == null) ? 0 : filterFunction.hashCode());
		result = prime * result + Arrays.hashCode(parameters);
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
		Filter other = (Filter) obj;
		if (filterFunction != other.filterFunction)
			return false;
		if (!Arrays.equals(parameters, other.parameters))
			return false;
		return true;
	}
	

	public static Filter returnExistingFilter(ESamples filterFunction2, Double[] params) {
		
			for(Entry<FilterIndex, Filter> entry : FilterLinkage.getFilters().entrySet()){
				Filter filter = entry.getValue();
				FilterIndex idx = entry.getKey();
				if(filterFunction2 != ESamples.RAND && filter.getFilterFunction() == filterFunction2 && Arrays.equals(filter.getParameters(), params)){
					System.out.println("Layer : " + idx.getId().getLayerId() + "  Area : " + idx.getId().getAreaId() +"  Filter : " + filter.id + "  " + filter.filterFunction + " réutilisé . " + Arrays.deepToString(params) + "  =  " + Arrays.deepToString(filter.getParameters()));
					return filter;
				}
			}
		return null;
	}

	public void resizeFilter() {
		
		int newWidth = maxX - minX + 1;
		int newHeight = maxY - minY + 1;
		
		if(newWidth <= 0 || newHeight <= 0){
			this.width = 0;
			this.height = 0;
			return;
		}
		
		Double[][] newValues = new Double[newWidth][newHeight];
		
		for(int idy = minY; idy <= maxY; idy++){
			for(int idx = minX; idx <= maxX; idx++){
				newValues[idx-minX][idy-minY] = values[idx][idy];
			}
			
		}
		
		this.values = newValues;
		this.width = newWidth;
		this.height = newHeight;
		
	}

	public Integer getWidth() {
		return width;
	}

	public void setWidth(Integer width) {
		this.width = width;
	}

	public Integer getHeight() {
		return height;
	}

	public void setHeight(Integer height) {
		this.height = height;
	}

}
