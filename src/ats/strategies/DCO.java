package ats.strategies;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

// Double cross over
public class DCO {

        boolean firstCross = true; // First cross over
        boolean goldenCross = false;
        boolean deadCross = false;

        public DCO() {

        }

        // Get double cross over
        public void getCrossOver(Instances instances, SMA sma, EMA ema) {

                // Create double cross over "DoubleCrossOver" attribute
                FastVector DoubleCrossOverList = new FastVector(3);
                DoubleCrossOverList.addElement("GoldenCross");
                DoubleCrossOverList.addElement("DeadCross");
                DoubleCrossOverList.addElement("NoCrossOver");

                Attribute DoubleCrossOver = new Attribute("DoubleCrossOver",
                                DoubleCrossOverList);
                instances.insertAttributeAt(DoubleCrossOver,
                                instances.numAttributes() - 1);

                double simpleMovingAverage = 0;
                double exponentialMovingAverage = 0;

                ArrayList<Double> SMASet = sma.getSMASet();
                ArrayList<Double> EMASet = ema.getEMASet();

                // Initialize "DoubleCrossOver" attribute
                instances.instance(0).setValue(instances.numAttributes() - 2, "NoCrossOver");

                // Determine double cross over
                for (int i = 1; i < instances.numInstances(); i++) {
                        if (SMASet.get(i) == EMASet.get(i)) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "NoCrossOver");
                        } else if ((SMASet.get(i) > EMASet.get(i)) && firstCross) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "GoldenCross");
                                firstCross = false;
                                goldenCross = true;
                                deadCross = false;
                        } else if ((SMASet.get(i) < EMASet.get(i)) && firstCross) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "DeadCross");
                                firstCross = false;
                                goldenCross = false;
                                deadCross = true;
                        } else if ((SMASet.get(i) > EMASet.get(i))
                                        && deadCross) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "GoldenCross");
                                goldenCross = true;
                                deadCross = false;
                        } else if ((SMASet.get(i) < EMASet.get(i))
                                        && goldenCross) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "DeadCross");
                                goldenCross = false;
                                deadCross = true;
                        } else {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "NoCrossOver");
                        }
                }
        }
}