import loadParametersP1
import loadFittingDataP1
import GradientDescent
import PlotGraph
import math
import SGD
import numpy as np
import Functions
import matplotlib
import time
import FiniteDifferenceApproximation
import random
import matplotlib.pyplot as plt
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 18})
    # PART 1:
    # =======
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1.getData()
    # GAUSSIAN:
    print "Q1.1 - Gaussian"
    # StepSizes = range(6, 15)
    # StepSize = math.exp(10)
    # Error = math.exp(-40)
    # Errors = range(-25, -2)
    # GaussGradPath, GaussOpt, GradNormGauss = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                          Functions.GradientGaussianFunc,
    #                                                                          [-70, 80], StepSize, Error,
    #                                                                          0, 0, 100000, gaussMean, gaussCov)
    # print "Gaussian - Function surface + path - Gaussian"
    # PlotGraph.PlotGraph3D("Gaussian Function and the Gradient Descent Path", Functions.GaussianFunc, GaussGradPath,
    #                       gaussMean, gaussCov)

    # # print "Gaussian - convergence of the gradient"
    # xes = range(0, len(GradNormGauss))
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, GradNormGauss)
    # plt.xlabel('Iteration')
    # plt.ylabel('Second Norm of the Gradient')
    # plt.title('Gradient Norm Convergence rate - Gaussian')
    # plt.savefig('Gradient Norm Convergence rate - Gaussian.png')
    # # print "Gaussian - convergence - different points"
    # ConvergenceRate = np.zeros((10, 10))
    # for x in range(0, 10):
    #     for y in range(0, 10):
    #         GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                             Functions.GradientGaussianFunc,
    #                                                                             [x, y], StepSize,
    #                                                                             Error, 0, 0, 100000, gaussMean,
    #                                                                             gaussCov)
    #         ConvergenceRate[x, y] = len(GradNorm)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Convergence for Different Starting Points - Gaussian')
    # plt.pcolor(ConvergenceRate)
    # plt.axis([0, 10, 0, 10])
    # plt.colorbar()
    # plt.savefig('Convergence for Different Starting Points - Gaussian.png')

    # # print "Gaussian - convergence - step size"
    # ConvergenceRate = []
    # for StepSize in StepSizes:
    #     print StepSize
    #     GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                         Functions.GradientGaussianFunc, [-10, 10],
    #                                                                         math.exp(StepSize),
    #                                                                         Error, 0, 0, 1000000, gaussMean, gaussCov)
    #     ConvergenceRate.append(len(GaussGradPath))
    # xes = StepSizes
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, ConvergenceRate)
    # plt.xlabel('ln(Learning Rate)')
    # plt.ylabel('Num of iterations to convergence')
    # plt.title('Convergence as a Function of the Learning Rate - Gaussian function')
    # # plt.show()
    #
    # # print"Gaussian - convergence - error"
    # ConvergenceRate = []
    # StepSize = math.exp(10)
    # for Error in Errors:
    #     GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                         Functions.GradientGaussianFunc, [-10, 10],
    #                                                                         StepSize,
    #                                                                         math.exp(Error), 0, 0, 1000000,
    #                                                                         gaussMean, gaussCov)
    #     ConvergenceRate.append(len(GaussGradPath))
    # xes = Errors
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, ConvergenceRate)
    # plt.xlabel('ln(Precision Level)')
    # plt.ylabel('Num of iterations to convergence')
    # plt.title('Convergence as a Function of the Precision Level - Gaussian function')

    #print "Gaussian - convergence - error and step size"
    # ConvergenceRate = np.zeros((len(Errors), len(StepSizes)))
    # for Error in range(0, len(Errors)):
    #     for StepSize in range(0, len(StepSizes)):
    #         print Error, ",", StepSize, ",", time.time()
    #         GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                             Functions.GradientGaussianFunc,
    #                                                                             [-10, 10],
    #                                                                             math.exp(StepSizes[StepSize]),
    #                                                                             math.exp(Errors[Error]), 0, 0, 1000000,
    #                                                                             gaussMean, gaussCov)
    #         ConvergenceRate[Error, StepSize] = len(GradNorm)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.xlabel('ln(Precision Level)')
    # plt.ylabel('ln(Learning Rate)')
    # plt.title('Convergence as a Function of the Precision Level and the Learning Rate - Gaussian')
    # plt.xticks(range(0, len(StepSizes)), StepSizes)
    # plt.yticks(range(0, len(Errors)), Errors)
    # plt.pcolor(ConvergenceRate)
    # print ConvergenceRate
    # plt.colorbar()
    # plt.savefig('Convergence as a Function of the Precision Level and the Learning Rate - Gaussian.png')


    #
    # print "Q1.1 - Quadratic Bowl"
    # Errors = range(-25, -1)
    # StepSizes = range(-13, -3)
    # StepSize = 0.1
    # Error = 0.00001
    # GaussGradPath, GaussOpt, GradNormBowl = GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                                                         Functions.GradientQuadraticBowlFunc,
    #                                                                         [-70, 80], StepSize, Error,
    #                                                                         0, 0, 100000, quadBowlA, quadBowlb)
    # print "Quadratic Bowl - Function surface + path"
    # PlotGraph.PlotGraph3D("Quadratic Bowl Function and the Gradient Descent Path", Functions.QuadraticBowlFunc,
    #                       GaussGradPath, quadBowlA, quadBowlb)
    #
    # # print "Quadratic Bowl - convergence of the gradient"
    # xes = range(0, len(GradNormBowl))
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, GradNormBowl)
    # plt.xlabel('Iteration')
    # plt.ylabel('Second Norm of the Gradient')
    # plt.title('Gradient Norm Convergence rate - Quadratic Bowl')
    # plt.savefig('Gradient Norm Convergence rate - Quadratic Bowl.png')
    #
    # # print "Quadratic Bowl - convergence - different points"
    # ConvergenceRate = np.zeros((10, 10))
    # for x in range(0, 10):
    #     for y in range(0, 10):
    #         GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                                                             Functions.GradientQuadraticBowlFunc,
    #                                                                             [x, y], StepSize, Error,
    #                                                                             0, 0, 100000, quadBowlA, quadBowlb)
    #         ConvergenceRate[x, y] = len(GradNorm)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Convergence for Different Starting Points - Quadratic Bowl')
    # plt.pcolor(ConvergenceRate)
    # plt.axis([0, 10, 0, 10])
    # plt.colorbar()
    # plt.savefig('Convergence for Different Starting Points - Quadratic Bowl.png')

    # # print "Quadratic Bowl - convergence - step size"
    # ConvergenceRate = []
    # for StepSize in StepSizes:
    #     print StepSize
    #     GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                                                         Functions.GradientQuadraticBowlFunc,
    #                                                                         [-10, 10], math.exp(StepSize),
    #                                                                         Error, 0, 0, 1000000, quadBowlA, quadBowlb)
    #     ConvergenceRate.append(len(GaussGradPath))
    # xes = StepSizes
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, ConvergenceRate)
    # plt.xlabel('ln(Learning Rate)')
    # plt.ylabel('Num of iterations to convergence')
    # plt.title('Convergence as a Function of the Learning Rate - Quadratic Bowl')
    #
    # # print "Quadratic Bowl - convergence - error"
    # ConvergenceRate = []
    # StepSize = 0.1
    # for Error in Errors:
    #     GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                                                         Functions.GradientQuadraticBowlFunc,
    #                                                                         [-10, 10], StepSize, math.exp(Error),
    #                                                                         0, 0, 1000000, quadBowlA, quadBowlb)
    #     ConvergenceRate.append(len(GaussGradPath))
    # xes = Errors
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, ConvergenceRate)
    # plt.xlabel('ln(Precision Level)')
    # plt.ylabel('Num of iterations to convergence')
    # plt.title('Convergence as a Function of the Precision Level - Quadratic Bowl')

    # # print "Gaussian - convergence - error and step size"
    # ConvergenceRate = np.zeros((len(Errors), len(StepSizes)))
    # for Error in range(0, len(Errors)):
    #     for StepSize in range(0, len(StepSizes)):
    #         print Error, ",", StepSize, ",", time.time()
    #         GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                                                             Functions.GradientQuadraticBowlFunc,
    #                                                                             [-10, 10],
    #                                                                             math.exp(StepSizes[StepSize]),
    #                                                                             math.exp(Errors[Error]), 0, 0, 1000000,
    #                                                                             quadBowlA, quadBowlb)
    #         ConvergenceRate[Error, StepSize] = len(GradNorm)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.xlabel('ln(Precision Level)')
    # plt.ylabel('ln(Learning Rate)')
    # plt.title('Convergence as a Function of the Precision Level and the Learning Rate - Quadratic Bowl')
    # plt.xticks(range(0, len(StepSizes)), StepSizes)
    # plt.yticks(range(0, len(Errors)), Errors)
    # plt.pcolor(ConvergenceRate)
    # print ConvergenceRate
    # plt.colorbar()
    # plt.savefig('Convergence as a Function of the Precision Level and the Learning Rate - Quadratic Bowl.png')
    #
    # #
    # #
    # #
    # #
    # # ##PART 2:
    # # ##=======
    x = y = np.arange(-100, 100, 0.1)
    GradApprox = []
    for i in range(0, 10000):
        SampledPointX = random.randint(0, len(x)-1)
        SampledPointY = random.randint(0, len(x)-1)
        Err = Functions.GradientGaussianFunc([x[SampledPointX], y[SampledPointY]], gaussMean, gaussCov) - \
                          FiniteDifferenceApproximation.FiniteDifferenceApproximation\
                          ([x[SampledPointX], y[SampledPointY]], 0.1, Functions.GaussianFunc, gaussMean, gaussCov)
        GradApprox.append(math.sqrt(np.dot(np.array(Err).T, np.array(Err))))
    counts, bin_edges = np.histogram(GradApprox, bins=100, normed=True)
    cdf = np.cumsum(counts)/counts.sum()
    counts.sum()
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel('Squared Error')
    plt.ylabel('Cumulative Amount')
    plt.title('Cumulative Gradient Error')
    plt.savefig('Cumulative Gradient Error.png')

    GradApprox=[]
    x=y=0
    for h in np.arange(0, 1, 0.1):
        Err = Functions.GradientGaussianFunc([x, y], gaussMean, gaussCov) - \
              FiniteDifferenceApproximation.FiniteDifferenceApproximation \
                  ([x, y], h, Functions.GaussianFunc, gaussMean, gaussCov)
        GradApprox.append(math.sqrt(np.dot(np.array(Err).T, np.array(Err))))

    plt.figure(figsize=(15, 10))
    plt.subplot()
    plt.plot( np.arange(0, 1, 0.1), GradApprox)
    plt.xlabel('h')
    plt.ylabel('Normed Error')
#    plt.title('Gradient Estimator SSE')
    plt.savefig('Gradient Estimator SSE.png')
    # # #
    # # #
    # # #

    ##PART 3:
    ##=======
    # Error=math.exp(-20)
    # Kappa=0.75
    # tau0=math.exp(25)
    # X,Y=loadFittingDataP1.getData()
    # StartingPoint=np.zeros(len(X[0]))
    # start_time = time.time()
    # SGDPath,SGDOpt=SGD.SGD(Functions.SSE,Functions.GradientSSEPoint,StartingPoint, tau0, Kappa, Error, 0,0,X, Y,10000000)
    # SGDElapsedTime=time.time() - start_time
    # start_time = time.time()
    # BatchPath,BatchOpt,gradpath=GradientDescent.GradientDescent(Functions.SSE,Functions.GradientSSE,StartingPoint,
    #                                                    (tau0**(-Kappa)),Error,0,0, 100000,X, Y)
    # BatchElapsedTime=time.time() - start_time
    # NumberOfIterSGD=len(SGDPath)/len(X)
    # NumberOfIterBatch=len(BatchPath)
    # print "Kappa: ",Kappa,"tau0: ",tau0
    #
    # print "Batch: running time is ", BatchElapsedTime, "Num of iterations is ",NumberOfIterBatch, "Final optimum is ",BatchOpt
    # print "SGD: running time is ", SGDElapsedTime, "Num of iterations is ", NumberOfIterSGD, "Final optimum is ", SGDOpt
    #


