public class Instance {
	
	private int label; // the category the instance belongs to
	private int[] x; // the data vector, x: N x 1

	public Instance(int[] x, int label) {
		this.x = x;
		this.label = label;
	}
	
	public Instance(int[] x) {
		this.x = x;
	}

	public int[] getX() {
		return x;
	}

	public void setX(int[] x) {
		this.x = x;
	}
	
	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}
	
}
