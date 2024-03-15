package ats;

import ats.classifiers.decisiontrees.BFDecisionTree;
import ats.classifiers.decisiontrees.C4_5DecisionTree;
import weka.classifiers.trees.J48;

public class ATS {

        public ATS() {
                super();
        }

        public void runATS() {

                // Generate training and test data sets
                DataSet dataSet = new DataSet();
                dataSet.generateDataSet();
                dataSet.generateTrainingDataSet();
                dataSet.generateTestDataSet();

                // Build classifiers
                C4_5DecisionTree c4_5DecisionTree = new C4_5DecisionTree(); // C4.5 decision tree
                c4_5DecisionTree.buildDecisionTree(dataSet.getTrainingDataSet());

                //BFDecisionTree bfDecisionTree = new BFDecisionTree(); // Best fit decision tree
                //bfDecisionTree.buildDecisionTree(dataSet.getTrainingDataSet());

                // Classify test data set
                c4_5DecisionTree.classify(dataSet.getTestDataSet()); // C4.5 decision tree
                //bfDecisionTree.classify(dataSet.getTestDataSet()); // Best fit decision tree

                // Evaluate classifier through cross validation
                c4_5DecisionTree.crossValidate(); // C4.5 decision tree
                //bfDecisionTree.crossValidate(); // Best fit decision tree

        }

        public static void main(String[] args) {

                ATS ats = new ATS();
                ats.runATS();

        }

}