import lassoData
import numpy as np
import sys
sys.path.append('../')
import BaseHandler


def Validate(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal,Lamdas,Ms,RgularizedFunction,type):
    BestW = []
    BestRSQ = float('Inf')
    BestLambda = 0
    BestM = 0
    for Lambda in Lamdas:
        for m in Ms:
            XBassedTrain = BaseHandler.BuildBasedData(DataSetXTrain,m,type)
            XBassedVal = BaseHandler.BuildBasedData(DataSetXVal,m,type)
            print XBassedTrain
            Inter,W,RSQ = RgularizedFunction(XBassedTrain, DataSetYTrain, XBassedVal, DataSetYVal, Lambda)
            # print "Lambda is ",Lambda, "M is", m, "RSQ is" ,RSQ
            if RSQ < BestRSQ:
                BestLambda = Lambda
                BestM = m
                BestRSQ = RSQ
    XBassedTrain = BaseHandler.BuildBasedData(DataSetXTrain, BestM, type)
    XBassedVal = BaseHandler.BuildBasedData(DataSetXVal, BestM, type)
    Inter, W, RSQ = RgularizedFunction(XBassedTrain, DataSetYTrain, XBassedVal, DataSetYVal, BestLambda)
    return Inter, W, BestLambda, BestM
