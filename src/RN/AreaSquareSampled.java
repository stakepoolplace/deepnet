package RN;

import java.util.List;

import RN.nodes.IPixelNode;
import RN.nodes.Node;

public class AreaSquareSampled extends AreaSquare implements IAreaSquare, IArea{
	
	private Integer sampling = 1;
	
	private AreaSquare sourceArea = null;
	
	private Integer nodeCount = null;
	
	
	
	public AreaSquareSampled(AreaSquare area, Integer sampling){
		
		this.sourceArea = area;
		this.sampling = sampling;
		this.nodeCount = (int) Math.pow(area.getWidthPx() / sampling, 2);
		
		initWidthPx(nodeCount);
		
	}
	
	public void initWidthPx(int pixSize){
		
		Double width = Math.sqrt(pixSize);
		
		if(width == 0D)
			throw new RuntimeException("Le nombre de neurones dans la liste est vide.");
		
		if(width.intValue() - width.doubleValue() != 0)
			throw new RuntimeException("Le nombre de neurones doit être un carré d'un entier.");
		
		// Carré
		this.widthPx = width.intValue();
		this.heightPx = this.widthPx;
		this.nodeCenterX = width.intValue() / 2;
		this.nodeCenterY = width.intValue() / 2;
	}


	public int getNodeCount() {
		return this.nodeCount;
	}

//	public List<INode> getNodes() {
//		
//		AreaSquareSampled area = this;
//		
//		Supplier<Stream<INode>> streamSupplier = () -> sourceArea.getNodes().stream().parallel()
//															.filter(node -> ((PixelNode) node).getX() % sampling == 0 )
//															.filter(node -> ((PixelNode) node).getY() % sampling == 0 );
//		
//		List<INode> nodes = streamSupplier.get().collect(Collectors.toList());
//		
//		streamSupplier.get().parallel().sequential().forEach(new Consumer<INode>(){
//			
//			int idx = 0;
//			
//			@Override
//			public void accept(INode node) {
//				node.setArea(area);
//				node.setNodeId(idx++);
//			}
//		});
//		
//		return nodes;
//	}
	
	
	@Override
	public IPixelNode getNodeCenterXY() {
		return sourceArea.getNodeCenterXY();
	}


	public IPixelNode getNodeXY(int x, int y) {
		x *= sampling;
		y *= sampling;
		
		return sourceArea.getNodeXY( x,  y);
	}

	public IPixelNode getNodeXY(int x, int y, int x0, int y0, double theta) {
		
		x *= sampling;
		y *= sampling;
		x0 *= sampling;
		y0 *= sampling;
		
		return sourceArea.getNodeXY( x,  y,  x0,  y0,  theta);
	}

	public List<IPixelNode> getNodesInSquareZone(int x0, int y0, int width, int height)  {
		
		x0 *= sampling;
		y0 *= sampling;
		
		return sourceArea.getNodesInSquareZone( x0,  y0,  width, height);
	}

	public List<IPixelNode> getNodesInCirclarZone(int x0, int y0, int radius) {
		
		x0 *= sampling;
		y0 *= sampling;
		radius *= sampling;
		
		return sourceArea.getNodesInCirclarZone( x0,  y0, radius);
	}
	
	
	public Integer getWidthPx() {
		return widthPx;
	}
	
	public Integer getHeightPx() {
		return heightPx;
	}


	public IArea getSourceArea() {
		return sourceArea;
	}

	public void setSourceArea(AreaSquare sourceArea) {
		this.sourceArea = sourceArea;
	}
	
	public int getX(Node node) {
		return nodeIdToNodeXY(node.getNodeId())[0];
	}
	
	public int getY(Node node) {
		return nodeIdToNodeXY(node.getNodeId())[1];
	}
	
	public int[] nodeIdToNodeXY(int id) {

		int x;
		int y;
		
		Double val = new Double(id / widthPx);
		y = val.intValue();
		x = id - y * widthPx;
		
		return new int[] { x, y };

	}
	


	
}
