package ats.classifiers.decisiontrees;

import java.util.ArrayList;
import java.util.Random;

import ats.DataSet;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.BFTree;
import weka.core.Instances;

// Best fit decision tree classifier 
public class BFDecisionTree {

        private BFTree decisionTree = null;

        public BFDecisionTree() {
                super();
        }

        // Build classifier
        public void buildDecisionTree(Instances trainingData) {

                try {
                        String[] options = new String[2];
                        options[0] = "-P <UNPRUNED>"; // Unpruned tree
                        options[0] = "-M <2>"; // The minimal number of instances at the terminal nodes
                        decisionTree = new BFTree(); 
                        decisionTree.setOptions(options);
                        // Train decision tree
                        decisionTree.buildClassifier(trainingData);
                } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                }
        }

        // Classify test data set
        public void classify(Instances testData) {

                ArrayList<double[]> classificationResultSet = new ArrayList<double[]>();
                double[] classificationResult = null;

                for (int i = 0; i < testData.numInstances(); i++) {

                        try {
                                classificationResult = decisionTree.distributionForInstance(testData.instance(i));
                        } catch (Exception e) {
                                // TODO Auto-generated catch block
                                e.printStackTrace();
                        }

                        classificationResultSet.add(i, classificationResult);
                }

                for (int i = 0; i < classificationResultSet.size(); i++) {
                        System.out.println(classificationResultSet.get(i).toString());
                }
        }

        // Evaluate classifier through cross validation
        public void crossValidate() {

                Instances newData = new Instances(DataSet.instances);

                try {
                        Evaluation eval = new Evaluation(newData);
                        BFTree untrainedDecitionTree = new BFTree();
                        eval.crossValidateModel(untrainedDecitionTree, newData, newData
                                        .numInstances(), new Random(1));
                        System.out.println(eval.toSummaryString("\nResults\n\n", false));
                } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                }
        }

}