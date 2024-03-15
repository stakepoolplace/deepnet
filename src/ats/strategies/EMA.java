package ats.strategies;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

// Exponential moving average
public class EMA {
        
        ArrayList<Double> EMASet = new ArrayList<Double>();

        public EMA() {
                super();
        }

        // Get exponential moving average
        public void getEMA(Instances instances, int EMADuration) {

                // Create exponential moving average "EMA" attribute
                FastVector EMAList = new FastVector(3);
                EMAList.addElement("rising");
                EMAList.addElement("falling");
                EMAList.addElement("flat");
                Attribute EMA = new Attribute("EMA" + EMADuration, EMAList);
                instances.insertAttributeAt(EMA, instances.numAttributes() - 1);

                if (instances.numInstances() >= EMADuration) {

                        double simpleMovingAverage = 0;
                        double exponentialMovingAverage = 0;
                        double sum = 0;
                        double multiplier = (double) (2 / ((double) EMADuration + 1)); // Weighting
                                                                                                                                                        // multiplier
                        double tempSum = 0;
                        double average = 0;

                        // Calculate simple "non-moving" average for
                        // the first "EMADuration - 1" instances
                        for (int i = 0; i < EMADuration - 1; i++) {
                                for (int j = 0; j <= i; j++) {
                                        tempSum += instances.instance(j).value(
                                                        instances.numAttributes() - 4);
                                        average = tempSum / (j + 1);
                                }
                                tempSum = 0;

                                // Record simple "non-moving" average
                                EMASet.add(i, average);

                                // Insert simple "non-moving" average into data set
                                if (i == 0) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "flat");
                                } else {
                                        if (EMASet.get(i) > EMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "rising");
                                        } else if (EMASet.get(i) < EMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "falling");
                                        } else if (EMASet.get(i) == EMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "flat");
                                        }
                                }
                        }

                        // Simple moving average of the first "EMADuration" instances
                        for (int i = 0; i < EMADuration; i++) {
                                sum += instances.instance(i).value(instances.numAttributes() - 4);
                        }
                        simpleMovingAverage = sum / EMADuration;
                        sum = 0;
                        // Record simple moving average of the first "EMADuration" instances
                        EMASet.add(EMADuration - 1, simpleMovingAverage);
                        simpleMovingAverage = 0;

                        // Calculate EMA
                        for (int i = EMADuration; i < instances.numInstances(); i++) {
                                exponentialMovingAverage = (instances.instance(i).value(
                                                instances.numAttributes() - 4) - EMASet.get(i - 1))
                                                * multiplier
                                                + EMASet.get(i - 1);
                                // Record EMA
                                EMASet.add(i, exponentialMovingAverage);
                        }

                        for (int i = EMADuration - 1; i < instances.numInstances(); i++) {
                                // Insert EMA into data set
                                if (EMASet.get(i) > EMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "rising");
                                } else if (EMASet.get(i) < EMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "falling");
                                } else if (EMASet.get(i) == EMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "flat");
                                }
                        }
                }
        }
        
        // Get numeric EMA values 
        public ArrayList<Double> getEMASet() {
                return EMASet;
        }
}