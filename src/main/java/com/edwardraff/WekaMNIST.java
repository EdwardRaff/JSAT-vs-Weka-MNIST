/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.edwardraff;

import java.io.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.neighboursearch.*;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Edward Raff
 */
public class WekaMNIST
{
    
    
    public static void main(String[] args) throws IOException, Exception
    {
        String folder = args[0];
        String trainPath = folder + "MNISTtrain.arff";
        String testPath = folder + "MNISTtest.arff";

        System.out.println("Weka Timings");
        Instances mnistTrainWeka = new Instances(new BufferedReader(new FileReader(new File(trainPath))));
        mnistTrainWeka.setClassIndex(mnistTrainWeka.numAttributes() - 1);
        Instances mnistTestWeka = new Instances(new BufferedReader(new FileReader(new File(testPath))));
        mnistTestWeka.setClassIndex(mnistTestWeka.numAttributes() - 1);
        
        //normalize range like into [0, 1]
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(mnistTrainWeka);
        
        mnistTestWeka = Normalize.useFilter(mnistTestWeka, normalizeFilter);
        mnistTrainWeka = Normalize.useFilter(mnistTrainWeka, normalizeFilter);
        
        long start, end;
        
        System.out.println("RBF SVM (Full Cache)");
        SMO smo = new SMO();
        smo.setKernel(new RBFKernel(mnistTrainWeka, 0/*0 causes Weka to cache the whole matrix...*/, 0.015625));
        smo.setC(8.0);
        smo.setBuildLogisticModels(false);
        evalModel(smo, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("RBF SVM (No Cache)");
        smo = new SMO();
        smo.setKernel(new RBFKernel(mnistTrainWeka, 1, 0.015625));
        smo.setC(8.0);
        smo.setBuildLogisticModels(false);
        evalModel(smo, mnistTrainWeka, mnistTestWeka);
        

        System.out.println("Decision Tree C45");
        J48 wekaC45 = new J48();
        wekaC45.setUseLaplace(false);
        wekaC45.setCollapseTree(false);
        wekaC45.setUnpruned(true);
        wekaC45.setMinNumObj(2);
        wekaC45.setUseMDLcorrection(true);

        evalModel(wekaC45, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("Random Forest 50 trees");
        int featuresToUse = (int) Math.sqrt(28 * 28);//Weka uses different defaults, so lets make sure they both use the published way

        RandomForest wekaRF = new RandomForest();
        wekaRF.setNumExecutionSlots(1);
        wekaRF.setMaxDepth(0/*0 for unlimited*/);
        wekaRF.setNumFeatures(featuresToUse);
        wekaRF.setNumTrees(50);
        
        evalModel(wekaRF, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("1-NN (brute)");
        IBk wekaNN = new IBk(1);
        wekaNN.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());
        wekaNN.setCrossValidate(false);
        
        evalModel(wekaNN, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("1-NN (Ball Tree)");
        wekaNN = new IBk(1);
        wekaNN.setNearestNeighbourSearchAlgorithm(new BallTree());
        wekaNN.setCrossValidate(false);
        
        evalModel(wekaNN, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("1-NN (Cover Tree)");
        wekaNN = new IBk(1);
        wekaNN.setNearestNeighbourSearchAlgorithm(new CoverTree());
        wekaNN.setCrossValidate(false);
        
        evalModel(wekaNN, mnistTrainWeka, mnistTestWeka);
        
        System.out.println("Logistic Regression LBFGS lambda = 1e-4");
        Logistic logisticLBFGS = new Logistic();
        logisticLBFGS.setRidge(1e-4);
        logisticLBFGS.setMaxIts(500);
        
        evalModel(logisticLBFGS, mnistTrainWeka, mnistTestWeka);
        
        
        System.out.println("k-means (Loyd)");
        int origClassIndex = mnistTrainWeka.classIndex();
        mnistTrainWeka.setClassIndex(-1);
        mnistTrainWeka.deleteAttributeAt(origClassIndex);
        {
            long totalTime = 0;
            for(int i = 0; i < 10; i++)
            {
                SimpleKMeans wekaKMeans = new SimpleKMeans();
                wekaKMeans.setNumClusters(10);
                wekaKMeans.setNumExecutionSlots(1);
                wekaKMeans.setFastDistanceCalc(true);

                start = System.currentTimeMillis();
                wekaKMeans.buildClusterer(mnistTrainWeka);
                end = System.currentTimeMillis();
                totalTime += (end-start);
            }
            System.out.println("\tClustering took: " + (totalTime/10.0) / 1000.0 + " on average");
        }
    }

    private static void evalModel(Classifier wekaModel, Instances train, Instances test) throws Exception
    {
        long start;
        long end;
        System.gc();
        start = System.currentTimeMillis();
        wekaModel.buildClassifier(train);
        end = System.currentTimeMillis();
        System.out.println("\tTraining took: " + (end - start) / 1000.0);
        
        System.gc();
        Evaluation eval = new Evaluation(train);
        start = System.currentTimeMillis();
        eval.evaluateModel(wekaModel, test);
        end = System.currentTimeMillis();
        System.out.println("\tEvaluation took " + (end - start) / 1000.0 + " seconds with an error rate " + eval.errorRate());
        
        System.gc();
    }
}
