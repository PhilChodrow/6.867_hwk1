from sklearn import linear_model
import numpy as np
import ValidateParams
import RegularizedFunctions
import lassoData
import matplotlib
import sys
sys.path.append('../')
import BaseHandler
import matplotlib.pyplot as plt
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 18})
    XTrain, YTrain = lassoData.lassoTrainData()
    XVal, YVal = lassoData.lassoValData()
    XTest,YTest=lassoData.lassoTestData()
    InterTrue=0.000000000000000000e+00
    wTrue=[ 0.000000000000000000e+00,5.646300000000000100e+00,7.785999999999999600e-01,
           0.000000000000000000e+00,8.108999999999999500e-01,2.682700000000000100e+00,0.000000000000000000e+00,
            0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00, 0.000000000000000000e+00,
            0.000000000000000000e+00 ]
    Lambdas=np.arange(0.1,10,0.5)
    Ms=range(0,10)
    InterLASSO,WLASSO, LambdaLASSO, MLASSO=ValidateParams.Validate(XTrain, YTrain, XVal, YVal,Lambdas,Ms,RegularizedFunctions.LASSO,"s")
    InterRIDGE,WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,Ms,
                                                       RegularizedFunctions.Ridge, "s")
    InterNoReg, WNoReg, LambdaRIDGE, MNoReg = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, [0], Ms,
                                                                      RegularizedFunctions.Ridge, "s")

    #plot the functions on the graph
    x = np.arange(-1, 1, 0.001)
    xBassed=BaseHandler.BuildBasedData(x,MLASSO,"s")
    yLASSO=InterLASSO+np.dot(np.array(xBassed),np.array(WLASSO))

    xBassed = BaseHandler.BuildBasedData(x, MRIDGE, "s")
    yRIDGE = InterRIDGE + np.dot(np.array(xBassed), np.array(WRIDGE))

    xBassed = BaseHandler.BuildBasedData(x, MNoReg, "s")
    yNoReg = InterNoReg + np.dot(np.array(xBassed), np.array(WNoReg))

    xBassed = BaseHandler.BuildBasedData(x, 12, "s")
    yTrue=InterTrue+ np.dot(np.array(xBassed), np.array(wTrue))
    plt.figure(figsize=(15, 10))
    plt.subplot()
    plt.plot(x, yLASSO, label='LASSO' )
    plt.plot(x, yRIDGE, label='Ridge')
    plt.plot(x, yTrue, label='True Function')
    plt.plot(x, yNoReg, label='Lambda = 0')
    plt.legend( loc=1, borderaxespad=0.)
    plt.plot(XTrain,YTrain,'o',label='Train Data')
    plt.plot(XVal, YVal,'s',label='Val Data')
    plt.plot(XTest,YTest, '^',label='Test Data')
    plt.savefig('Estimated Functions.png')

    # Plot w as a function of lambda:
    #Lambdas = np.arange(0.0001, 0.1, 0.00001)
    XBassedTrain = BaseHandler.BuildBasedData(XTrain, 5, "s")
    XBassedVal = BaseHandler.BuildBasedData(XVal, 5, "s")
    WLASSO=[]
    WRidge=[]
    plt.figure(figsize=(15, 10))
    plt.subplot()
    for Lambda in Lambdas:
        Inter,wLASSO, RSQ = RegularizedFunctions.LASSO(XBassedTrain, YTrain, XBassedVal, YVal, Lambda)
        WLASSO.append(wLASSO)
        Inter,wRidge, RSQ = RegularizedFunctions.Ridge(XBassedTrain, YTrain, XBassedVal, YVal, Lambda)
        WRidge.append(wRidge)
    for w in range(0,len(WLASSO[0])):
        lasso=[]
        ridge=[]
        for l in range(0,len(Lambdas)):
            lasso.append(WLASSO[l][w])
            ridge.append(WRidge[l][w])
        #print "debug"
        #print lasso
        plt.plot(Lambdas,lasso,'--')
        plt.plot(Lambdas, ridge, )
    plt.savefig('Ws as a function of Lambda.png')

    #plot weights
    InterLASSO, WLASSO, LambdaLASSO, MLASSO = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, [12],
                                                                      RegularizedFunctions.LASSO, "s")
    InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, [12],
                                                                      RegularizedFunctions.Ridge, "s")
    InterNoReg, WNoReg, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, [0], [12],
                                                                      RegularizedFunctions.Ridge, "s")
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.title('LASSO')
    plt.bar(range(0,len(wTrue)),WLASSO)
    plt.subplot(222)
    plt.title('Ridge')
    plt.bar(range(0, len(wTrue)), WRIDGE)
    plt.subplot(223)
    plt.title('True')
    plt.bar(range(0, len(wTrue)), wTrue)
    plt.subplot(224)
    plt.title('Lambda = 0')
    plt.bar(range(0, len(wTrue)), WNoReg)
    plt.savefig('Different Ws.png')
