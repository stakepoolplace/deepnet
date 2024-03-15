package ats.strategies;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

// Relative Strength Index
public class RSI {

        ArrayList<Double> rsi = new ArrayList<Double>();

        public RSI() {
                super();
        }

        // Get Relative Strength Index
        public void getRSI(Instances instances, int period) {

                // Create Relative Strength Index "RSI" attribute
                FastVector RSIList = new FastVector(3);
                RSIList.addElement("overbought");
                RSIList.addElement("oversold");
                RSIList.addElement("neither");
                Attribute RSI = new Attribute("RSI", RSIList);
                instances.insertAttributeAt(RSI, instances.numAttributes() - 1);

                ArrayList<Double> change = new ArrayList<Double>();
                ArrayList<Double> gain = new ArrayList<Double>();
                ArrayList<Double> loss = new ArrayList<Double>();
                ArrayList<Double> averageGain = new ArrayList<Double>();
                ArrayList<Double> averageLoss = new ArrayList<Double>();
                double gainSum = 0;
                double lossSum = 0;

                // Initialize
                for (int i = 0; i < instances.numInstances(); i++) {
                        change.add(0.0);
                        gain.add(0.0);
                        loss.add(0.0);
                        averageGain.add(0.0);
                        averageLoss.add(0.0);
                        rsi.add(0.0);
                }

                // Calculate "change", "gain" and "loss"
                for (int i = 1; i < instances.numInstances(); i++) {
                        change
                                        .set(i, instances.instance(i).value(
                                                        instances.numAttributes() - 7)
                                                        - instances.instance(i - 1).value(
                                                                        instances.numAttributes() - 7));
                        if (change.get(i) > 0) {
                                gain.set(i, change.get(i));
                                loss.set(i, 0.0);
                        } else if (change.get(i) < 0) {
                                loss.set(i, change.get(i) * (-1.0));
                                gain.set(i, 0.0);
                        }
                }

                // Calculate first "average gain" and "average loss"
                for (int i = 0; i < period; i++) {
                        gainSum += gain.get(i);
                }
                for (int i = 0; i < loss.size(); i++) {
                        lossSum += loss.get(i);
                }
                averageGain.set(period, gainSum / ((double) period));
                averageLoss.set(period, lossSum / ((double) period));

                // Calculate subsequent "average gain" and "average loss" values
                for (int i = period + 1; i < instances.numInstances(); i++) {
                        averageGain.set(i, (averageGain.get(i - 1)
                                        * ((double) (period - 1)) + gain.get(i))
                                        / ((double) period));
                        averageLoss.set(i, (averageLoss.get(i - 1)
                                        * ((double) (period - 1)) + loss.get(i))
                                        / ((double) period));
                }

                // Calculate RSI
                double RS = 0; // Relative strength
                double RSIndex = 0; // Relative strength index
                for (int i = period; i < instances.numInstances(); i++) {
                        RS = averageGain.get(i) / averageLoss.get(i);
                        RSIndex = 100 - 100 / (1 + RS);
                        rsi.set(i, RSIndex);
                }

                // Set "RSI" attribute
                for (int i = period; i < instances.numInstances(); i++) {
                        if (rsi.get(i) > 70.0) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "overbought");
                        } else if (rsi.get(i) < 30.0) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "oversold");
                        } else if (rsi.get(i) >= 30 && rsi.get(i) <= 70) {
                                instances.instance(i).setValue(instances.numAttributes() - 2,
                                                "neither");
                        }
                }
        }

        // Get numeric RSI values
        public ArrayList<Double> getRSI() {
                return rsi;
        }
}