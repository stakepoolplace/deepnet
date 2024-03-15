package RN;

public class Identification {
	
	private Integer layerId = null;
	
	private Integer areaId = null;
	
	private Integer nodeId = null;

	public Identification(Integer layerId) {
		this.layerId = layerId;
	}
	
	public Identification(Integer layerId, Integer areaId) {
		this.layerId = layerId;
		this.areaId = areaId;
	}
	
	public Identification(Integer layerId, Integer areaId, Integer nodeId) {
		this.layerId = layerId;
		this.areaId = areaId;
		this.nodeId = nodeId;
	}
	
	public String getIdentification() {
		return " L[" + layerId + "] A[" + areaId + "] N[" + nodeId + "]";
	}
	
	public String toString(){
		return getIdentification();
	}

	public Integer getLayerId() {
		return layerId;
	}

	public void setLayerId(Integer layerId) {
		this.layerId = layerId;
	}

	public Integer getAreaId() {
		return areaId;
	}

	public void setAreaId(Integer areaId) {
		this.areaId = areaId;
	}

	public Integer getNodeId() {
		return nodeId;
	}

	public void setNodeId(Integer nodeId) {
		this.nodeId = nodeId;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((areaId == null) ? 0 : areaId.hashCode());
		result = prime * result + ((layerId == null) ? 0 : layerId.hashCode());
		result = prime * result + ((nodeId == null) ? 0 : nodeId.hashCode());
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
		Identification other = (Identification) obj;
		
		if (areaId == null) {
			if (other.areaId != null)
				return false;
		} else if (!areaId.equals(other.areaId))
			return false;
		if (layerId == null) {
			if (other.layerId != null)
				return false;
		} else if (!layerId.equals(other.layerId))
			return false;
		if (nodeId == null) {
			if (other.nodeId != null)
				return false;
		} else if (!nodeId.equals(other.nodeId))
			return false;
		return true;
	}
	
	

}
