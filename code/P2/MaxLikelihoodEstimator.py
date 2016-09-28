import sys
sys.path.append('../')
import BaseHandler
import numpy as np
import numpy.linalg as linalg
def EstimateMaxLikelihoodCoeff(X,Y,M,type):
    XBassed=BaseHandler.BuildBasedData(X,M,type)
    Phi=np.array(XBassed)
    W_ML=np.dot(np.dot(linalg.inv(np.dot(Phi.T, Phi)), Phi.T), Y)
    return W_ML

def EstimateFunction(X,W,M,type):
    Xbassed=BaseHandler.BuildBasedData([X],M,type)
    return np.dot(np.array(W).T, Xbassed[0])

