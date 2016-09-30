from sklearn import linear_model
import numpy as np
import ValidateParams
import RegularizedFunctions
import lassoData
import matplotlib
import math
import sys
sys.path.append('../')
import BaseHandler
import matplotlib.pyplot as plt
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 18})
    XTrain, YTrain = lassoData.lassoTrainData()
    XVal, YVal = lassoData.lassoValData()
    XTest,YTest=lassoData.lassoTestData()
    wTrue=[ 0.000000000000000000e+00,0.000000000000000000e+00,5.646300000000000100e+00,7.785999999999999600e-01,
           0.000000000000000000e+00,8.108999999999999500e-01,2.682700000000000100e+00,0.000000000000000000e+00,
            0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00, 0.000000000000000000e+00,
            0.000000000000000000e+00 ]
    Lambdas=np.arange(0.001,5,0.05)
    Ms=[13]
    InterLASSO,WLASSO, LambdaLASSO, MLASSO=ValidateParams.Validate(XTrain, YTrain, XVal, YVal,Lambdas,Ms,RegularizedFunctions.LASSO,"s")
    InterRIDGE,WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,Ms,
                                                       RegularizedFunctions.Ridge, "s")
    Ms=[4]
    InterNoReg, WNoReg, Lambdanoreg, MNoReg = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, [0], Ms,
                                                                      RegularizedFunctions.Ridge, "s")

    print "lasso:",LambdaLASSO,"ridge:",LambdaRIDGE
    print InterNoReg
    #plot the functions on the graph
    x = np.arange(-1, 1, 0.001)
    xBassed=BaseHandler.BuildBasedData(x,MLASSO,"s")
    yLASSO=InterLASSO+np.dot(np.array(xBassed),np.array(WLASSO))

    xBassed = BaseHandler.BuildBasedData(x, MRIDGE, "s")
    yRIDGE = InterRIDGE + np.dot(np.array(xBassed), np.array(WRIDGE))

    xBassed = BaseHandler.BuildBasedData(x, MNoReg, "s")
    yNoReg =  np.dot(np.array(xBassed), np.array(WNoReg))

    xBassed = BaseHandler.BuildBasedData(x, 13, "s")
    yTrue= np.dot(np.array(xBassed), np.array(wTrue))
    plt.figure(figsize=(15, 10))
    plt.subplot()
    plt.plot(x, yLASSO, label='LASSO' )
    plt.plot(x, yRIDGE, label='Ridge')
    plt.plot(x, yTrue, label='True Function')
    plt.plot(x, yNoReg, label='Lambda = 0')
    plt.legend( loc=4, borderaxespad=0.)
    plt.plot(XTrain,YTrain,'o',label='Train Data')
    plt.plot(XVal, YVal,'s',label='Val Data')
    plt.plot(XTest,YTest, '^',label='Test Data')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Estimation of the function for the optimal parameters')
    plt.savefig('Estimated Functions.png')

    # Plot w as a function of lambda:
    # Lambdas = np.arange(0.01, 5, 0.01)
    # XBassedTrain = BaseHandler.BuildBasedData(XTrain, 5, "s")
    # XBassedVal = BaseHandler.BuildBasedData(XVal, 5, "s")
    # WLASSO=[]
    # WRidge=[]
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # for Lambda in Lambdas:
    #     Inter,wLASSO, RSQ = RegularizedFunctions.LASSO(XBassedTrain, YTrain, XBassedVal, YVal, Lambda)
    #     WLASSO.append(wLASSO)
    #     Inter,wRidge, RSQ = RegularizedFunctions.Ridge(XBassedTrain, YTrain, XBassedVal, YVal, Lambda)
    #     WRidge.append(wRidge)
    # for w in range(0,len(WLASSO[0])):
    #     lasso=[]
    #     ridge=[]
    #     for l in range(0,len(Lambdas)):
    #         lasso.append(WLASSO[l][w])
    #         ridge.append(WRidge[l][w])
    #     #print "debug"
    #     #print lasso
    #     plt.plot(Lambdas,lasso)
    #     plt.plot(Lambdas, ridge,'--' )
    #     plt.gca().semilogx()
    # plt.xlabel('Lambda')
    # plt.ylabel('w')
    # #plt.title('W as a function of Lambda')
    # plt.savefig('Ws as a function of Lambda.png')

    #plot weights
    InterLASSO, WLASSO, LambdaLASSO, MLASSO = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, [13],
                                                                      RegularizedFunctions.LASSO, "s")
    InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, [13],
                                                                      RegularizedFunctions.Ridge, "s")
    InterNoReg, WNoReg, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, [0], [13],
                                                                      RegularizedFunctions.Ridge, "s")
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.title('(a)')
    plt.bar(range(0,len(wTrue)),WLASSO)
    plt.subplot(222)
    plt.title('(b)')
    plt.bar(range(0, len(wTrue)), WRIDGE)
    plt.subplot(223)
    plt.title('(c)')
    plt.bar(range(0, len(wTrue)), wTrue)
    plt.subplot(224)
    plt.title('(d)')
    plt.bar(range(0, len(wTrue)), WNoReg)
    plt.savefig('Different Ws.png')
