package ats.classifiers.kernelmachines;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import ats.DataSet;

// RBF support vector classifier
public class RBFSupportVector {
        
        private RBFKernel rbfSupportVector = null;
        
        public RBFSupportVector() {
                super();
        }
        
        // Build classifier
        public void buildRBFSupportVector(Instances trainingData) {

                try {
                        String[] options = new String[2];
                        options[0] = "-C 250007"; // Cache size
                        options[1] = "-G 0.01"; // Gamma parameter
                        rbfSupportVector = new RBFKernel(); // new instance of RBF support vector
                        rbfSupportVector.setOptions(options);
                        // Train decision tree
                        rbfSupportVector.buildKernel(trainingData);
                } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                }
        }

        // Classify test data set
        public void classify(Instances testData) {

                ArrayList<Double> classificationResultSet = new ArrayList<Double>();
                Double classificationResult = null;

                for (int i = 0; i < testData.numInstances(); i++) {

                        try {
                                //classificationResult = rbfSupportVector.classifyInstance(testData
                                                //.get(i));
                        } catch (Exception e) {
                                // TODO Auto-generated catch block
                                e.printStackTrace();
                        }

                        classificationResultSet.add(i, classificationResult);
                }

                for (int i = 0; i < classificationResultSet.size(); i++) {
                        System.out.println(classificationResultSet.get(i));
                }
        }

        // Evaluate classifier through cross validation
        public void crossValidate() {

                Instances newData = new Instances(DataSet.instances);

                try {
                        Evaluation eval = new Evaluation(newData);
                        RBFKernel untrainedDecitionTree = new RBFKernel();
                        //eval.crossValidateModel(untrainedDecitionTree, newData, newData
                                        //.numInstances(), new Random(1));
                        System.out.println(eval.toSummaryString("\nResults\n\n", false));
                } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                }
        }
        
        

}