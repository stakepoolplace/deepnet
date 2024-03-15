package RN.nodes;

import RN.IArea;
import RN.IAreaSquare;
import RN.ITester;
import RN.algoactivations.EActivation;
import RN.dataset.Coordinate;
import RN.linkage.vision.Gradient;
import RN.links.Link;

public class PixelNode extends Node implements IPixelNode, INode{
	
	private Coordinate coordinate = new Coordinate();
	
	private Gradient gradient = null;

	public PixelNode() {
		super(EActivation.IDENTITY);
		this.nodeType = ENodeType.PIXEL;
	}

	public PixelNode(EActivation activationFx) {
		super(activationFx);
		this.nodeType = ENodeType.PIXEL;
	}
	
	public PixelNode(INode innerNode) {
		super(innerNode);
		this.nodeType = ENodeType.PIXEL;
		this.activationFx = EActivation.IDENTITY;
		this.activationFxPerNode = true;
	}
	

	public void initParameters(){
	}
	
	public IArea getArea() {
		return  super.getArea();
	}
	
	
	public int getX() {
		
		if(coordinate.getX() == null){
			coordinate.setX((double) getAreaSquare().getX(this));
			coordinate.setX0((double) getAreaSquare().getNodeCenterX());
		}
		
		if(coordinate.getY() == null){
			coordinate.setY((double) getAreaSquare().getY(this));
			coordinate.setY0((double) getAreaSquare().getNodeCenterY());
		}
		
		return coordinate.getX().intValue();
	}


	public int getY() {
		
		if(coordinate.getX() == null){
			coordinate.setX((double) getAreaSquare().getX(this));
			coordinate.setX0((double) getAreaSquare().getNodeCenterX());
		}
		
		if(coordinate.getY() == null){
			coordinate.setY((double) getAreaSquare().getY(this));
			coordinate.setY0((double) getAreaSquare().getNodeCenterY());
		}
		
		return coordinate.getY().intValue();
	}
	
	public double getR() {
		
		if(coordinate.getR() == null){
			if(coordinate.getX() == null || coordinate.getY() == null){
				getX();
				getY();
			}
			coordinate.linearToPolarSystem();
		}
		
		return coordinate.getR();
	}
	
	
	public double getP(double base) {
		
		if(coordinate.getP() == null){
			if(coordinate.getX() == null || coordinate.getY() == null){
				getX();
				getY();
			}
			coordinate.setBase(base);
			coordinate.linearToLogPolarSystem();
		}
		
		return coordinate.getP();
	}
	
	
	public double getTheta() {
		
		if(coordinate.getTheta() == null){
			if(coordinate.getX() == null || coordinate.getY() == null){
				getX();
				getY();
			}
				
			coordinate.linearToPolarSystem();
		}
		
		return coordinate.getTheta();
	}
	
	
	public void setTheta(Double theta){
		coordinate.setTheta(theta);
	}

	public void setP(Double p){
		coordinate.setP(p);
	}
	
	public void setR(Double r){
		coordinate.setR(r);
	}

	@Override
	public IPixelNode getLeft() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x - 1, y);
		} catch (Exception ignore) {
			System.err.println("getLeft(" + (x-1) +"," + y +") not accessible () "); 
		}
		
		return null;
	}
	

	@Override
	public IPixelNode getRight() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x + 1, y);
		} catch (Exception ignore) {
			System.err.println("getRight(" + (x+1) +"," + y +") not accessible () "); 
		}
		
		return null;
	}

	@Override
	public IPixelNode getUp() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x, y - 1);
		} catch (Exception ignore) {
			System.err.println("getUp(" + x +"," + (y-1) +") not accessible () "); 
		}
		
		return null;
		
	}
	
	@Override
	public IPixelNode getUpLeft() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x - 1, y - 1);
		} catch (Exception ignore) {
			System.err.println("getUpLeft(" + (x-1) +"," + (y-1) +") not accessible () "); 
		}
		
		return null;
	}
	
	@Override
	public IPixelNode getUpRight() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x + 1, y - 1);
		} catch (Exception ignore) {
			System.err.println("getUpRight(" + (x+1) +"," + (y-1) +") not accessible () "); 
		}
		
		return null;
	}

	@Override
	public IPixelNode getDown() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x, y + 1);
		} catch (Exception ignore) {
			System.err.println("getDown(" + x +"," + (y+1) +") not accessible () "); 
		}
		
		return null;
	}
	
	@Override
	public IPixelNode getDownLeft() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x - 1, y + 1);
		} catch (Exception ignore) {
			System.err.println("getDownLeft(" + (x-1) +"," + y +") not accessible () "); 
		}
		
		return null;
	}
	
	@Override
	public IPixelNode getDownRight() {
		int x = getAreaSquare().getX(this);
		int y = getAreaSquare().getY(this);
		try {
			return getAreaSquare().getNodeXY(x + 1, y + 1);
		} catch (Exception ignore) {
			System.err.println("getLeft(" + (x-1) +"," + y +") not accessible () "); 
		}
		
		return null;
	}
	


	@Override
	public String getString() {
		String result = "            NODE : " + nodeId + " type : " + this.nodeType;
		result += ITester.NEWLINE + "             x : " + getAreaSquare().getX(this) + "   y :" + getAreaSquare().getY(this) ;
		result += ITester.NEWLINE + "                  INPUTS : " + (inputs.isEmpty() ? " _" : "");
		for (Link link : inputs) {
			result += ITester.NEWLINE + "                            " + link.getString();
		}
		result += ITester.NEWLINE + "                    BIAS : " + ITester.NEWLINE + "                            "
				+ (biasInput != null ? biasInput.getString() : "aucun");
		// result += "\n                  ACTIVATION FX : " + this.function +
		// "\n                  OUTPUT :\n                      " + this.output;
		result += ITester.NEWLINE + "                   ERROR : " + ITester.NEWLINE + "                            " + this.error;
		result += ITester.NEWLINE + "                  OUTPUT : ";
		for (Link link : outputs) {
			result += ITester.NEWLINE + "                            " + link.getString();
		}
		result += ITester.NEWLINE;
		return result;
	}
	
	@Override
	public String toString() {
		return "PixelNode [id=" + nodeId +", layer=" + (area != null ? area.getLayer().getLayerId() : "") + ", area=" + (area != null ? area.getAreaId() : "") + ", x=" + getX() + ", y=" + getY() + ", output=" + computedOutput + "]";
	}

	@Override
	public IAreaSquare getAreaSquare() {
		return (IAreaSquare) area;
	}
	
	@Override
	public Double compareOutputTo(IPixelNode comparedNode) {
		return this.getComputedOutput() - comparedNode.getComputedOutput();
	}
	
	@Override
	public int compareOutputTo(Double value) {
		
		if(value == null || this.getComputedOutput() > value)
			return 1;
		else if(this.getComputedOutput() < value)
			return -1;
		
		return 0;
	}
	
	public IAreaSquare getNextAreaSquare() {
		
		try{
			return (IAreaSquare) area.getLayer().getArea(area.getAreaId() + 1);
		}catch(Throwable ignore){
		}
		
		return null;
	}
	
	public IAreaSquare getPreviousAreaSquare() {
		try{
			return (IAreaSquare) area.getLayer().getArea(area.getAreaId() - 1);
		}catch(Throwable ignore){
		}
		
		return null;
	}
	
	

	
	private Gradient computeGradient() {
		
		try{
			double magnitude = Math.sqrt(
					Math.pow(getLeft().getComputedOutput() - getRight().getComputedOutput() , 2D) +
					Math.pow(getDown().getComputedOutput() - getUp().getComputedOutput() , 2D)
					);
			
			double theta = Math.atan2(
					(getUp().getComputedOutput() - getDown().getComputedOutput()),
					(getLeft().getComputedOutput() - getRight().getComputedOutput())
					);
			
			if(theta < 0D)
				theta += (2D * Math.PI);
			
			
			return new Gradient(this, theta, magnitude, null);
			
			
		}catch(Throwable ignore){
			//ignore.printStackTrace();
		}
		
		return null;
		
	}
	
	public Double distance(IPixelNode n1, IPixelNode n2){
		
		return Math.sqrt(Math.pow(n1.getX() - n2.getX(), 2D) + Math.pow(n1.getY() - n2.getY(), 2D));
		
	}
	
	public Double distance(IPixelNode n){
		return distance(this, n);
	}

	public Gradient getGradient() {
		if(gradient == null){
			this.gradient = computeGradient();
		}
		return gradient;
	}

	public Coordinate getCoordinate() {
		return coordinate;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
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
		Node other = (Node) obj;
		if (nodeId == null) {
			if (other.nodeId != null)
				return false;
		} else if (!nodeId.equals(other.nodeId))
			return false;
		return true;
	}





}
