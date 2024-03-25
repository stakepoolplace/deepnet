package RN;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import RN.dataset.InputData;
import RN.dataset.OutputData;
import RN.nodes.ENodeType;
import javafx.collections.FXCollections;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;

public class GlobalNetwork {
	
	private static GlobalNetwork instance = null;
	
	private static List<INetwork> networks = new ArrayList<INetwork>();

	public List<LineChart.Series<Number, Number>> seriesFirstNETInputs = new ArrayList<LineChart.Series<Number, Number>>();
	public List<LineChart.Series<Number, Number>> seriesFirstNETOutputs = new ArrayList<LineChart.Series<Number, Number>>();
	public List<LineChart.Series<Number, Number>> seriesLastNETInputs = new ArrayList<LineChart.Series<Number, Number>>();
	public List<LineChart.Series<Number, Number>> seriesLastNETOutputs = new ArrayList<LineChart.Series<Number, Number>>();
	public LineChart<Number, Number> sc = ViewerFX.lineChart;
	
	public static GlobalNetwork getInstance() {
		if (instance == null) {
			instance = new GlobalNetwork();
		}

		return instance;
	}
	

	public void launchRealCompute() throws Exception {

		
		DataSeries dataSeries = DataSeries.getInstance();
		int idxNetwork = 0;
		INetwork network = get(idxNetwork);

		if (network != null && !dataSeries.getInputDataSet().isEmpty()) {

			OutputData output = null;

			if (sc.getData() == null)
				sc.setData(FXCollections.<XYChart.Series<Number, Number>> observableArrayList());

			LineChart.Series<Number, Number> series = null;
			seriesFirstNETInputs = new ArrayList<LineChart.Series<Number, Number>>();
			seriesFirstNETOutputs = new ArrayList<LineChart.Series<Number, Number>>();
			seriesLastNETInputs = new ArrayList<LineChart.Series<Number, Number>>();
			seriesLastNETOutputs = new ArrayList<LineChart.Series<Number, Number>>();

			for (int idx = 0; idx < network.getFirstLayer().getLayerNodes(ENodeType.REGULAR).size(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run in" + (sc.getData().size() + 1));
				seriesFirstNETInputs.add(series);
			}

			for (int idx = 0; idx < network.getLastLayer().getNodeCount(); idx++) {
				series = new LineChart.Series<Number, Number>();
				series.setName("Run out[" + idx + "]" + (sc.getData().size() + 1));
				seriesFirstNETOutputs.add(series);
			}
			


			int runCycle = 0;
			ListIterator computeItr = dataSeries.getInputDataSet().listIterator();
			for (InputData entry : dataSeries.getInputDataSet()) {
				int idx = 0;

				for (LineChart.Series<Number, Number> seriesIn : seriesFirstNETInputs) {
					seriesIn.getData().add(new LineChart.Data<Number, Number>(runCycle, entry.getInput(idx++)));
				}
				
				try {
					output = network.compute(computeItr);
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				idx = 0;
				for (LineChart.Series<Number, Number> seriesOut : seriesFirstNETOutputs) {
					seriesOut.getData().add(new LineChart.Data<Number, Number>(runCycle, output.getOutput(idx++)));
				}
				
				while(++idxNetwork < size() - 1){
					network = get(idxNetwork);
					output = network.propagation(false);
				}
				
				if(seriesLastNETInputs.size() == 0)
					for (idx = 0; idx < network.getFirstLayer().getLayerNodes(ENodeType.REGULAR).size(); idx++) {
						series = new LineChart.Series<Number, Number>();
						series.setName("Run in" + (sc.getData().size() + 1));
						seriesLastNETInputs.add(series);
					}
				
				idx = 0;
				for (LineChart.Series<Number, Number> seriesIn : seriesLastNETInputs) {
					seriesIn.getData().add(new LineChart.Data<Number, Number>(runCycle, entry.getInput(idx++)));
				}

				if(seriesLastNETOutputs.size() == 0)
				for (idx = 0; idx < network.getLastLayer().getNodeCount(); idx++) {
					series = new LineChart.Series<Number, Number>();
					series.setName("Run out[" + idx + "]" + (sc.getData().size() + 1));
					seriesLastNETOutputs.add(series);
				}
				
				idx = 0;
				for (LineChart.Series<Number, Number> seriesOut : seriesLastNETOutputs) {
					seriesOut.getData().add(new LineChart.Data<Number, Number>(runCycle, output.getOutput(idx++)));
				}


				
				if (ViewerFX.showLogs.isSelected()) {
					System.out.print("[  ");
					for (double input : entry.getInput()) {
						System.out.print(input + "  ");
					}
					System.out.print("]");

					for (double out : output.getOutput()) {
						System.out.print(", [actual=" + out);
//						System.out.print(", ideal=" + entry.getIdeal(idx++) + "]");
					}
					System.out.println(" ");
				}

				runCycle++;
				
				
			}
		
		}
	}


	public int size() {
		return networks.size();
	}


	public boolean isEmpty() {
		return networks.isEmpty();
	}


	public boolean contains(Object o) {
		return networks.contains(o);
	}


	public Iterator<INetwork> iterator() {
		return networks.iterator();
	}


	public Object[] toArray() {
		return networks.toArray();
	}


	public <T> T[] toArray(T[] a) {
		return networks.toArray(a);
	}


	public boolean add(INetwork e) {
		return networks.add(e);
	}


	public boolean remove(Object o) {
		return networks.remove(o);
	}


	public boolean containsAll(Collection<?> c) {
		return networks.containsAll(c);
	}


	public boolean addAll(Collection<? extends INetwork> c) {
		return networks.addAll(c);
	}


	public boolean addAll(int index, Collection<? extends INetwork> c) {
		return networks.addAll(index, c);
	}


	public boolean removeAll(Collection<?> c) {
		return networks.removeAll(c);
	}


	public boolean retainAll(Collection<?> c) {
		return networks.retainAll(c);
	}


	public void clear() {
		networks.clear();
	}


	public boolean equals(Object o) {
		return networks.equals(o);
	}


	public int hashCode() {
		return networks.hashCode();
	}


	public INetwork get(int index) {
		return networks.get(index);
	}


	public INetwork set(int index, INetwork element) {
		return networks.set(index, element);
	}


	public void add(int index, INetwork element) {
		networks.add(index, element);
	}


	public INetwork remove(int index) {
		return networks.remove(index);
	}


	public int indexOf(Object o) {
		return networks.indexOf(o);
	}


	public int lastIndexOf(Object o) {
		return networks.lastIndexOf(o);
	}


	public ListIterator<INetwork> listIterator() {
		return networks.listIterator();
	}


	public ListIterator<INetwork> listIterator(int index) {
		return networks.listIterator(index);
	}


	public List<INetwork> subList(int fromIndex, int toIndex) {
		return networks.subList(fromIndex, toIndex);
	}
	
}
