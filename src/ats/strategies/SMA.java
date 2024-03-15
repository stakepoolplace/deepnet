package ats.strategies;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

// Simple moving average
public class SMA {
        
        ArrayList<Double> SMASet = new ArrayList<Double>();

        public SMA() {
                super();
        }

        // Get simple moving average
        public void getSMA(Instances instances, int SMADuration) {

                // Create simple moving average "SMA" attribute
                FastVector SMAList = new FastVector(3);
                SMAList.addElement("rising");
                SMAList.addElement("falling");
                SMAList.addElement("flat");
                Attribute SMA = new Attribute("SMA", SMAList);
                instances.insertAttributeAt(SMA, instances.numAttributes() - 1);

                if (instances.numInstances() >= SMADuration) {

                        double simpleMovingAverage = 0;
                        double sum = 0;
                        double tempSum = 0;
                        double average = 0;

                        // Calculate simple "non-moving" average for
                        // the first "SMADuration - 1" instances
                        for (int i = 0; i < SMADuration - 1; i++) {
                                for (int j = 0; j <= i; j++) {
                                        tempSum += instances.instance(j).value(
                                                        instances.numAttributes() - 3);
                                        average = tempSum / (j + 1);
                                }
                                tempSum = 0;

                                // Record simple "non-moving" average
                                SMASet.add(i, average);

                                // Insert simple "non-moving" average into data set
                                if (i == 0) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "flat");
                                } else {
                                        if (SMASet.get(i) > SMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "rising");
                                        } else if (SMASet.get(i) < SMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "falling");
                                        } else if (SMASet.get(i) == SMASet.get(i - 1)) {
                                                instances.instance(i).setValue(
                                                                instances.numAttributes() - 2, "flat");
                                        }
                                }
                        }

                        // Calculate SMA
                        for (int i = SMADuration - 1; i < instances.numInstances(); i++) {
                                for (int j = 0; j < SMADuration; j++) {
                                        sum += instances.instance(i - j).value(
                                                        instances.numAttributes() - 3);
                                }
                                simpleMovingAverage = sum / SMADuration;
                                sum = 0;

                                // Record SMA
                                SMASet.add(i, simpleMovingAverage);

                                // Insert SMA into data set
                                if (SMASet.get(i) > SMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "rising");
                                } else if (SMASet.get(i) < SMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "falling");
                                } else if (SMASet.get(i) == SMASet.get(i - 1)) {
                                        instances.instance(i).setValue(instances.numAttributes() - 2,
                                                        "flat");
                                }

                        }
                }
        }
        
        // Get numeric SMA values 
        public ArrayList<Double> getSMASet() {
                return SMASet;
        }
}