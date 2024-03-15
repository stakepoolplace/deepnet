package ats.classifiers.decisiontrees;

import java.util.ArrayList;
import java.util.Random;

import ats.DataSet;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

// C4.5 decision tree classifier 
public class C4_5DecisionTree {

        private J48 decisionTree = null;

        public C4_5DecisionTree() {
                super();
        }

        // Build classifier
        public void buildDecisionTree(Instances trainingData) {

                try {
                        String[] options = new String[1];
                        options[0] = "-U"; // unpruned tree
                        decisionTree = new J48(); // new instance of tree
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

                ArrayList<Double> classificationResultSet = new ArrayList<Double>();
                Double classificationResult = null;

                for (int i = 0; i < testData.numInstances(); i++) {

                        try {
                                classificationResult = decisionTree.classifyInstance(testData
                                                .instance(i));
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
                        J48 untrainedDecitionTree = new J48();
                        eval.crossValidateModel(untrainedDecitionTree, newData, newData
                                        .numInstances(), new Random(1));
                        System.out.println(eval.toSummaryString("\nResults\n\n", false));
                } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                }
        }

}