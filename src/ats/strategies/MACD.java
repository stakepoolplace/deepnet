package ats.strategies;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

// Moving Average Convergence-Divergence
public class MACD {

        ArrayList<Double> MACDSet = new ArrayList<Double>();

        public MACD() {
                super();
        }

        public void getMACD(Instances instances, EMA ema) {

                // Create Moving Average Convergence-Divergence "MACD" attribute
                FastVector MACDList = new FastVector(2);
                MACDList.addElement("positive");
                MACDList.addElement("negative");
                MACDList.addElement("centerline");
                Attribute MACD = new Attribute("MACD", MACDList);
                instances.insertAttributeAt(MACD, instances.numAttributes() - 1);

                // Get 12-Day EMA
                EMA EMA_12Day = new EMA();
                EMA_12Day.getEMA(instances, 12);
                instances.deleteAttributeAt(instances.numAttributes() - 2);
                ArrayList<Double> EMA_12DaySet = EMA_12Day.getEMASet();

                // Get 26_Day EMA
                EMA EMA_26Day = new EMA();
                EMA_26Day.getEMA(instances, 26);
                instances.deleteAttributeAt(instances.numAttributes() - 2);
                ArrayList<Double> EMA_26DaySet = EMA_26Day.getEMASet();

                // Calculate MACD
                for (int i = 0; i < EMA_12DaySet.size(); i++) {
                        MACDSet.add(i, EMA_12DaySet.get(i) - EMA_26DaySet.get(i));
                }

                // Set "MACD" attribute
                for (int i = 0; i < instances.numInstances(); i++) {
                        if (MACDSet.get(i) > 0) {
                                instances.instance(i).setValue(instances.instance(i).numAttributes() - 2,
                                                "positive");
                        } else if (MACDSet.get(i) < 0) {
                                instances.instance(i).setValue(instances.instance(i).numAttributes() - 2,
                                                "negative");
                        } else if (MACDSet.get(i) == 0) {
                                instances.instance(i).setValue(instances.instance(i).numAttributes() - 2,
                                                "centerline");
                        }
                }

        }
}