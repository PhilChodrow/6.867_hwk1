import sys
sys.path.append('../P4')
sys.path.append('../')
import BaseHandler
import matplotlib.pyplot as plt
import numpy as np
import ValidateParams
import RegularizedFunctions
import lassoData
import matplotlib
import loadFittingDataP2
import regressData
import matplotlib

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 18})
    ## Part 1:

    X, Y = loadFittingDataP2.getData(ifPlotData=False)
    plt.figure(figsize=(15, 10))
    plt.subplot()
    xes= np.arange(0, 1, 0.0001)
    # Plot for different values of Lambda and M

    #Lambdas=[0.00001,0.5,1]
    Lambdas = [0.001,0.1,0.2]
    Ms=[2,4,7]
    for Lambda in Lambdas:
        for m in Ms:
            XBassed = BaseHandler.BuildBasedData(X,m,"p")
            Inter,W,RSQVal = RegularizedFunctions.Ridge(XBassed, Y, XBassed, Y, Lambda)
            XesBassed=BaseHandler.BuildBasedData(xes,m,"p")
            y = Inter+np.dot(np.array(XesBassed), np.array(W))
            Lable='M= '+str(m)+' Lambda= '+str(Lambda)
            plt.plot(xes, y, label=Lable)
    plt.plot(X,Y,'o',label='Data')
    plt.legend(loc=1, borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Estimation of the function for different Parameters')
    plt.savefig('Estimation of the function for different Parameters Part1.png')


    #X, Y = loadFittingDataP2.getData(ifPlotData=False)
    # InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, Ms,
    #                                                                   RegularizedFunctions.Ridge, "p")

    ## Part 2:
    # Get best lambda and
    Lambdas=np.arange(0.1,10,0.5)
    Ms=range(2,10)
    XA,YA=regressData.regressAData()
    XB,YB=regressData.regressBData()
    XVal,YVal=regressData.validateData()
    InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XA,YA, XVal, YVal, Lambdas, Ms,
                                                                      RegularizedFunctions.Ridge, "p")
    XBassedTrain = BaseHandler.BuildBasedData(XA, MRIDGE, "p")
    XBassedVal = BaseHandler.BuildBasedData(XVal, MRIDGE, "p")
    XBassedTest = BaseHandler.BuildBasedData(XB, MRIDGE, "p")
    Inter, W, RSQ_AVAL = RegularizedFunctions.Ridge(XBassedTrain,YA, XBassedVal, YVal,LambdaRIDGE)
    Inter, W, RSQ_AB = RegularizedFunctions.Ridge(XBassedTrain, YA, XBassedTest, YB, LambdaRIDGE)
    print "M ",MRIDGE,"LAMBDA " ,Lambda, "SSE", RSQ_AB

    plt.figure(figsize=(15, 10))
    plt.subplot()
    xes= np.arange(-3, 2.5, 0.0001)
    # Plot for different values of Lambda and M
    Lambdas=[0.00001,0.5]
    Ms=[2,5,9]
    for Lambda in Lambdas:
        for m in Ms:
            XBassedTrain = BaseHandler.BuildBasedData(XA,m,"p")
            XBassedVal = BaseHandler.BuildBasedData(XVal,m,"p")
            XBassedTest = BaseHandler.BuildBasedData(XB,m,"p")
            Inter,W,RSQVal = RegularizedFunctions.Ridge(XBassedTrain, YA, XBassedVal, YVal, Lambda)
            Inter, W, RSQTest = RegularizedFunctions.Ridge(XBassedTrain, YA, XBassedTest, YB, Lambda)
            print "Lambda ",Lambda,"M ",m,"RSQTest ",RSQTest
            XesBassed=BaseHandler.BuildBasedData(xes,m,"p")
            y = Inter+np.dot(np.array(XesBassed), np.array(W))
            Lable='M= '+str(m)+' Lambda= '+str(Lambda)
            plt.plot(xes, y, label=Lable)
    plt.plot(XA,YA,'o',label='Set A')
    plt.plot(XB,YB,'s',label='Set B')
    plt.plot(XVal,YVal, '^',label='Val Data')
    plt.legend(loc=4, borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Estimation of the function for different Parameters')
    plt.savefig('Estimation of the function for different Parameters abval.png')