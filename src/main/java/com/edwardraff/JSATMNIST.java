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

import java.io.File;
import java.util.*;
import jsat.ARFFLoader;
import jsat.classifiers.*;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.linear.*;
import jsat.classifiers.linear.kernelized.KernelSGD;
import jsat.classifiers.svm.DCDs;
import jsat.classifiers.svm.PlatSMO;
import jsat.classifiers.svm.SupportVectorLearner;
import jsat.classifiers.trees.*;
import jsat.clustering.SeedSelectionMethods;
import jsat.clustering.kmeans.*;
import jsat.datatransform.*;
import jsat.datatransform.kernel.RFF_RBF;
import jsat.distributions.kernels.KernelPoint;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.*;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.*;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.SoftmaxLoss;

/**
 *
 * @author Edward Raff
 */
public class JSATMNIST
{
    public static void main(String[] args)
    {
        String folder = args[0];
        String trainPath = folder + "MNISTtrain.arff";
        String testPath = folder + "MNISTtest.arff";
        
        ClassificationDataSet mnistTrainJSAT = new ClassificationDataSet(ARFFLoader.loadArffFile(new File(trainPath)), 0);
        ClassificationDataSet mnistTestJSAT = new ClassificationDataSet(ARFFLoader.loadArffFile(new File(testPath)), 0);
        
        System.out.println("JSAT Timings");
        
        System.out.println("RBF SVM (Full Cache)");
        PlatSMO smo = new PlatSMO(new RBFKernel(5.65685));
        smo.setC(8.0);
        smo.setTolerance(1e-3);
        smo.setCacheMode(SupportVectorLearner.CacheMode.FULL);
        evaluate(new OneVSOne(smo), mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory());
        
        System.out.println("RBF SVM (No Cache)");
        smo = new PlatSMO(new RBFKernel(5.65685));
        smo.setC(8.0);
        smo.setTolerance(1e-3);
        smo.setCacheMode(SupportVectorLearner.CacheMode.NONE);
        evaluate(new OneVSOne(smo), mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory());
        
        System.out.println("RBF SVM stochastic w/ 5 iterations");
        KernelSGD ksgd = new KernelSGD(new HingeLoss(), new RBFKernel(5.65685), 1/(2*mnistTrainJSAT.getSampleSize()*8.0), KernelPoint.BudgetStrategy.MERGE_RBF, 1000);
        ksgd.setEpochs(5);
        evaluate(ksgd, mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory());
        
        System.out.println("RBF SVM RKS features w/ Linear Solver");
        DCDs dcds = new DCDs();
        dcds.setC(8.0);
        dcds.setUseL1(true);
        evaluate(new OneVSOne(dcds), mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory(), new RFF_RBF.RFF_RBFTransformFactory(5.65685, 1000, true));
        
        System.out.println("Decision Tree C45");
        DecisionTree jsatC45 = DecisionTree.getC45Tree();
        jsatC45.setMinResultSplitSize(2);
        jsatC45.setMinSamples(2);
        jsatC45.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN_RATIO);
        jsatC45.setPruningMethod(TreePruner.PruningMethod.NONE);
        evaluate(jsatC45, mnistTrainJSAT, mnistTestJSAT);
        
        System.out.println("Random Forest 50 trees");
        RandomForest rf = new RandomForest(50);
        evaluate(rf, mnistTrainJSAT, mnistTestJSAT);
        
        System.out.println("1-NN (brute)");
        NearestNeighbour nnBrute = new NearestNeighbour(1, new VectorArray.VectorArrayFactory<VecPaired<Vec,Double>>());
        evaluate(nnBrute, mnistTrainJSAT, mnistTestJSAT);
        
        System.out.println("1-NN (VPmv)");
        NearestNeighbour nnVPmv = new NearestNeighbour(1, new VPTreeMV.VPTreeMVFactory<VecPaired<Vec,Double>>(VPTree.VPSelection.Random));
        evaluate(nnVPmv, mnistTrainJSAT, mnistTestJSAT);
        
        System.out.println("1-NN (Random Ball Cover)");
        NearestNeighbour nnRBC = new NearestNeighbour(1, new RandomBallCover.RandomBallCoverFactory<VecPaired<Vec,Double>>());
        evaluate(nnRBC, mnistTrainJSAT, mnistTestJSAT);
        
        System.out.println("Log Regression stochastic w/ 10 iterations");
        LinearSGD logRegSGD = new LinearSGD(new SoftmaxLoss(), 1e-4, 0.0);
        logRegSGD.setEpochs(10);
        evaluate(logRegSGD, mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory());
        
        System.out.println("Logistic Regression OneVsAll DCD");
        LogisticRegressionDCD jsatDCD = new LogisticRegressionDCD(1.0);
        jsatDCD.setUseBias(true);
        jsatDCD.setMaxIterations(100000);
        evaluate(new OneVSAll(jsatDCD, false), mnistTrainJSAT, mnistTestJSAT, new LinearTransform.LinearTransformFactory());
        
        System.out.println("Logistic Regression LBFGS lambda = 1e-4");
        LinearBatch logRegExact = new LinearBatch(new SoftmaxLoss(), 1e-4);
        evaluate(logRegExact, mnistTrainJSAT, mnistTestJSAT);
        
        //k-means
        long start, end, total;
        total = 0;
        for(int i = 0; i < 10; i++)
        {
            NaiveKMeans jsatModel = new NaiveKMeans();
            start = System.currentTimeMillis();
            jsatModel.cluster(mnistTrainJSAT, 10);
            end = System.currentTimeMillis();
            total += (end - start);
        }
        System.out.println("Loyd kMeans Time: " + (total / 10.0) / 1000.0 + " on average");

        total = 0;
        for (int i = 0; i < 10; i++)
        {
            HamerlyKMeans jsatModel2 = new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.KPP);
            start = System.currentTimeMillis();
            jsatModel2.cluster(mnistTrainJSAT, 10);
            end = System.currentTimeMillis();
            total += (end - start);
        }

        System.out.println("Hamerly kMeans Time: " + (total / 10.0) / 1000.0 + " on average");

        total = 0;
        for (int i = 0; i < 10; i++)
        {
            KMeans jsatModel3 = new ElkanKMeans(new EuclideanDistance(), new Random(), SeedSelectionMethods.SeedSelection.KPP);
            start = System.currentTimeMillis();
            jsatModel3.cluster(mnistTrainJSAT, 10);
            end = System.currentTimeMillis();
            total += (end - start);
        }

        System.out.println("Elkan kMeans Time: " + (total / 10.0) / 1000.0 + " on average");
    }

    private static void evaluate(Classifier classifier, ClassificationDataSet mnistTrainJSAT, ClassificationDataSet mnistTestJSAT, DataTransformFactory... transforms)
    {
        System.gc();
        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(classifier, mnistTrainJSAT);
        if(transforms != null && transforms.length > 0)
            cme.setDataTransformProcess(new DataTransformProcess(transforms));
        cme.evaluateTestSet(mnistTestJSAT);
        System.out.println("\tTraining took: " + cme.getTotalTrainingTime() / 1000.0);
        System.out.println("\tEvaluation took: " + cme.getTotalClassificationTime() / 1000.0 + " seconds with an error rate " + cme.getErrorRate());
        System.gc();
    }
}
