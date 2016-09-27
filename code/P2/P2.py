import matplotlib.pyplot as plt
import loadFittingDataP2
import MaxLikelihoodEstimator
import time
import math
import numpy as np
import sys
sys.path.append('../P1')
import GradientDescent
import SGD
import Functions
if __name__ == '__main__':
    ##PART 1:
    ##=======
    X,Y=loadFittingDataP2.getData(ifPlotData=False)
    xes=np.arange(0.0, 1, 0.0001)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X, Y, 'o')
    fun=[]
    for m in range(1,10):
        W_ML=MaxLikelihoodEstimator.EstimateMaxLikelihoodCoeff(X,Y,m,"p")
        fun=[]
        for j in xes:
            fun.append(MaxLikelihoodEstimator.EstimateFunction(j,W_ML,m,"p"))
        plt.plot(xes, fun)
    plt.show()

    ##PART 2:
    ##=======
    #implemented in Q1

    ##PART 3:
    ##=======
    Error = math.exp(-20)
    Kappa = 0.75
    tau0 = math.exp(10)
    StartingPoint = 0
    start_time = time.time()
    SGDPath, SGDOpt = SGD.SGD(Functions.SSE,Functions.GradientSSEPoint,StartingPoint, tau0, Kappa, Error,0,0, X, Y)
    SGDElapsedTime = time.time() - start_time
    start_time = time.time()
    BatchPath, BatchOpt = GradientDescent.GradientDescent(Functions.SSE, Functions.GradientSSE, StartingPoint,
                                                          (tau0 ** (-Kappa)), Error, 0,0,X, Y)
    BatchElapsedTime = time.time() - start_time
    NumberOfIterSGD = len(SGDPath) / len(X)
    NumberOfIterBatch = len(BatchPath)
    print "Batch: running time is ", BatchElapsedTime, "Num of iterations is ", NumberOfIterBatch, "Final optimum is ", BatchOpt
    print "SGD: running time is ", SGDElapsedTime, "Num of iterations is ", NumberOfIterSGD, "Final optimum is ", SGDOpt

    ##PART 4:
    ##=======
    plt.figure(2)
    plt.subplot(211)
    plt.plot(X, Y, 'o')
    fun = []
    for m in range(1, 10):
        W_ML = MaxLikelihoodEstimator.EstimateMaxLikelihoodCoeff(X, Y, m, "c")
        fun = []
        for j in xes:
            fun.append(MaxLikelihoodEstimator.EstimateFunction(j, W_ML, m, "c"))
        plt.plot(xes, fun)
    plt.show()