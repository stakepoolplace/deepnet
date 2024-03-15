package ats.classifiers.bayesiannetworks;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierTemplate extends Classifier {

        public ClassifierTemplate() {
                super();
        }

        // Generates classifier
        @Override
        public void buildClassifier(Instances arg0) throws Exception {
                // TODO Auto-generated method stub

        }

        // Classifies the given test instance
        @Override
        public double classifyInstance(Instance arg0) throws Exception {
                // TODO Auto-generated method stub
                return 0;
        }

        // Predicts the class membership of a given instance
        @Override
        public double[] distributionForInstance(Instance arg0) throws Exception {
                // TODO Auto-generated method stub
                return null;
        }

        // Returns classifier capabilities
        @Override
        public Capabilities getCapabilities() {
                // TODO Auto-generated method stub
                return null;
        }

}