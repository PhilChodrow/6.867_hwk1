import loadParametersP1
import loadFittingDataP1
import GradientDescent
import PlotGraph
import math
import SGD
import numpy as np
import Functions
import time
import FiniteDifferenceApproximation
import matplotlib.pyplot as plt
if __name__ == '__main__':
    ##PART 1:
    ##=======
    [gaussMean,gaussCov,quadBowlA,quadBowlb]=loadParametersP1.getData()
    #GAUSSIAN:
    StepSize=math.exp(10)
    Error=math.exp(-250)
    print "Starting with the gaussian function"
    GaussGradPath,GaussOpt = GradientDescent.GradientDescent(Functions.GaussianFunc,
                                                            Functions.GradientGaussianFunc,[-10,10],StepSize,
                                                            Error,0,0,gaussMean,gaussCov)
    ##Function serface + path - Gaussian
    #PlotGraph.PlotGraph3D("Gaussian", Functions.GaussianFunc,GaussGradPath,gaussMean,gaussCov)
    #
    # # GAUSSIAN - convergence - step size
    # ConvergenceRate = []
    # for StepSize in range(1,10):
    #     GaussGradPath, GaussOpt = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                               Functions.GradientGaussianFunc, [-10, 10], math.exp(StepSize),
    #                                                               Error, 0,0,gaussMean, gaussCov)
    #     ConvergenceRate.append(len(GaussGradPath))
    # ##Convergence as a function of the stepsize - Gaussian
    # xes=range(1,10)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(xes, ConvergenceRate)
    #
    #
    # # GAUSSIAN - convergence - error
    # ConvergenceRate = []
    # StepSize = math.exp(10)
    # for Error in range(1, 10):
    #     GaussGradPath, GaussOpt = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                               Functions.GradientGaussianFunc, [-10, 10],
    #                                                               StepSize,
    #                                                               math.exp(Error), 0,0,gaussMean, gaussCov)
    #     ConvergenceRate.append(len(GaussGradPath))
    # ##Convergence as a function of the error - Gaussian
    # xes=range(1,10)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(xes, ConvergenceRate)
    #
    #
    # StepSize = 0.1
    # Error = 0.00001
    # print "Quadratic Bowl"
    # QuadraticBowlGradPath, QuadraticBowlOpt=GradientDescent.GradientDescent(Functions.QuadraticBowlFunc,
    #                                           Functions.GradientQuadraticBowlFunc,[20,-10],StepSize,Error,0,0,quadBowlA,quadBowlb)
    # print QuadraticBowlOpt
    # print QuadraticBowlGradPath
    # ##Function serface + path - Quadratic Bowl
    # PlotGraph.PlotGraph3D("Quadratic Bowl",Functions.QuadraticBowlFunc, QuadraticBowlGradPath, quadBowlA,quadBowlb)


    ##PART 2:
    ##=======
    point=[5,5]
    step=math.exp(-5)
    print "Gaussian:\n==============\n"
    print "for point"
    print point
    print "Gradient value is:"
    print Functions.GradientGaussianFunc(point,gaussMean,gaussCov)
    print "Finite Difference Gradient approximation:"
    print FiniteDifferenceApproximation.FiniteDifferenceApproximation(point,step,Functions.GaussianFunc,gaussMean,gaussCov)
    print "Quadratic Bowl:\n==============\n"
    print "for point"
    print point
    print "Gradient value is:"
    print Functions.GradientQuadraticBowlFunc(point,quadBowlA,quadBowlb)
    print "Finite Difference Gradient approximation:"
    print FiniteDifferenceApproximation.FiniteDifferenceApproximation(point,step,Functions.QuadraticBowlFunc,quadBowlA,quadBowlb)

    ##PART 3:
    ##=======
    Error=math.exp(-20)
    Kappa=0.75
    tau0=math.exp(20)
    X,Y=loadFittingDataP1.getData()
    StartingPoint=np.zeros(len(X[0]))
    start_time = time.time()
    SGDPath,SGDOpt=SGD.SGD(Functions.SSE,Functions.GradientSSEPoint,StartingPoint, tau0, Kappa, Error, 0,0,X, Y)
    SGDElapsedTime=time.time() - start_time
    start_time = time.time()
    BatchPath,BatchOpt=GradientDescent.GradientDescent(Functions.SSE,Functions.GradientSSE,StartingPoint,(tau0**(-Kappa)),Error,0,0, X, Y)
    BatchElapsedTime=time.time() - start_time
    NumberOfIterSGD=len(SGDPath)/len(X)
    NumberOfIterBatch=len(BatchPath)
    print "Batch: running time is ", BatchElapsedTime, "Num of iterations is ",NumberOfIterBatch, "Final optimum is ",BatchOpt
    print "SGD: running time is ", SGDElapsedTime, "Num of iterations is ", NumberOfIterSGD, "Final optimum is ", SGDOpt



