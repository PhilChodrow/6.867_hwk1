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
import FiniteDifferenceApproximation
import random

if __name__ == '__main__':
    # PART 1:
    # =======
    X, Y = loadFittingDataP2.getData(ifPlotData=False)
    # xes = np.arange(0.0, 1, 0.0001)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(X, Y, 'o', label='Data')
    # fun = []
    # for m in range(1, 10, 2):
    #     W_ML = MaxLikelihoodEstimator.EstimateMaxLikelihoodCoeff(X, Y, m, "p")
    #     fun = []
    #     for j in xes:
    #         fun.append(MaxLikelihoodEstimator.EstimateFunction(j, W_ML, m, "p"))
    #     plt.plot(xes, fun, label='M='+str(m))
    # RealW = [0, 1, 1.5]
    # fun = []
    # for j in xes:
    #     fun.append(MaxLikelihoodEstimator.EstimateFunction(j, RealW, 2, "c"))
    # plt.plot(xes, fun, linewidth=2.5, label='Real Function')
    # plt.legend( loc=1, borderaxespad=0.)
    # plt.xlabel('x')
    # plt.ylabel('y(x)')
    # plt.title('Polynomial Basis Maximum Likelihood Estimator')
    # plt.savefig('Polynomial Basis Maximum Likelihood Estimator.png')
    #
    # # PART 2:
    # # =======
    # implemented in Q1
    plt.figure(figsize=(15, 10))
    theta = range(0, 100)
    GradApprox = []
    for i in range(0, 1000000):
        SampledPointTHETA = random.randint(0, len(theta)-1)
        Err = Functions.GradientSSE(theta[SampledPointTHETA], X, Y) - \
                          FiniteDifferenceApproximation.FiniteDifferenceApproximation\
                          (theta[SampledPointTHETA], 0.1, Functions.SSE, X, Y)
        GradApprox.append(math.sqrt(np.dot(np.array(Err).T, np.array(Err))))
    counts, bin_edges = np.histogram(GradApprox, bins=100, normed=True)
    cdf = np.cumsum(counts)/counts.sum()
    print np.mean(np.cumsum(counts)/counts.sum())
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel('Squared Error')
    plt.ylabel('Cumulative Amount')
    plt.title('Cumulative Gradient Error')
    plt.savefig('Cumulative Gradient Error.png')
    #
    #
    # # # PART 3:
    # # # =======
    # StartingPoint=0
    # StepSizes = range(-25, -5)
    # Errors = range(-20, 0)
    # Tau0s = range(0, 15)
    # BatchConvergenceRate = np.zeros((len(Errors), len(StepSizes)))
    # SGDConvergenceRate = np.zeros((len(Errors), len(Tau0s)))
    # for Error in range(0, len(Errors)):
    #     for StepSize in range(0, len(StepSizes)):
    #         print Error, ",", StepSize, ",", time.time()
    #         BatchGradPath, BatchOpt, BatchGradNorm = GradientDescent.GradientDescent(Functions.SSE,
    #                                                                                  Functions.GradientSSE,
    #                                                                                  StartingPoint,
    #                                                                                  math.exp(StepSizes[StepSize]),
    #                                                                                  math.exp(Errors[Error]), 0, 0,
    #                                                                                  1000000, X, Y)
    #         BatchConvergenceRate[Error, StepSize] = len(BatchGradNorm)
    #     for tau0 in range(0,len(Tau0s)):
    #         BatchGradPath, BatchOpt = SGD.SGD(Functions.SSE,
    #                                                          Functions.GradientSSEPoint,
    #                                                          StartingPoint,
    #                                                          math.exp(Tau0s[tau0]),
    #                                                          0.75,
    #                                                          math.exp(Errors[Error]), 0, 0,
    #                                                          X, Y)
    #         SGDConvergenceRate[Error, tau0] = len(BatchGradPath)
    #
    # # plt.figure(figsize=(15, 10))
    # # plt.subplot()
    # # plt.xlabel('ln(Precision Level)')
    # # plt.ylabel('ln(Learning Rate)')
    # # plt.title('Convergence as a Function of the Precision Level and the Learning Rate - Batchl')
    # # plt.xticks(range(0, len(StepSizes)), StepSizes)
    # # plt.yticks(range(0, len(Errors)), Errors)
    # # plt.pcolor(BatchConvergenceRate)
    # # plt.colorbar()
    # # plt.savefig('Convergence as a Function of the Precision Level and the Learning Rate - Batch.png')
    #
    # # plt.figure(figsize=(15, 10))
    # # plt.subplot()
    # # plt.xlabel('ln(Precision Level)')
    # # plt.ylabel('ln(Learning Rate)')
    # # plt.title('Convergence as a Function of the Precision Level and the Learning Rate - SGD')
    # # plt.xticks(range(0, len(Tau0s)), Tau0s)
    # # plt.yticks(range(0, len(Errors)), Errors)
    # # plt.pcolor(SGDConvergenceRate)
    # # plt.colorbar()
    # # plt.savefig('Convergence as a Function of the Precision Level and the Learning Rate - SGD.png')
    #
    # # PART 4:
    # =======
    xes = np.arange(0.0, 1, 0.0001)
    plt.figure(figsize=(15, 10))
    plt.subplot()
    plt.plot(X, Y, 'o', label='Data')
    fun = []
    for m in range(1, 8):
        W_ML = MaxLikelihoodEstimator.EstimateMaxLikelihoodCoeff(X, Y, m, "c")
        fun = []
        for j in xes:
            fun.append(MaxLikelihoodEstimator.EstimateFunction(j, W_ML, m, "c"))
        plt.plot(xes, fun, label='M='+str(m))
    RealW = [0, 1, 1.5]
    fun = []
    for j in xes:
        fun.append(MaxLikelihoodEstimator.EstimateFunction(j, RealW, 2, "c"))
    plt.plot(xes, fun, linewidth=2.5, label='Real Function')
    plt.legend( loc=1, borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Cosin Basis Maximum Likelihood Estimator')
    plt.savefig('Cosin Basis Maximum Likelihood Estimator.png')

