import sys
sys.path.append('../P4')
sys.path.append('../P2')
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

if __name__ == '__main__':
    #X, Y = loadFittingDataP2.getData(ifPlotData=False)
    # InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas, Ms,
    #                                                                   RegularizedFunctions.Ridge, "p")


    Lambdas=np.arange(0.1,10,0.5)
    Ms=range(0,10)
    XA,YA=regressData.regressAData()
    XB,YB=regressData.regressBData()
    XVal,YVal=regressData.validateData()
    InterRIDGE, WRIDGE, LambdaRIDGE, MRIDGE = ValidateParams.Validate(XA,YA, XVal, YVal, Lambdas, Ms,
                                                                      RegularizedFunctions.Ridge, "p")