import LASSO
import sys
import BaseHandler
import lassoData
import numpy as np
sys.path.append('../')

if __name__ == '__main__':
    XTrain, YTrain = lassoData.lassoTrainData()
    XVal, YVal = lassoData.lassoTrainData()
    print YVal
    XBassedTrain = BaseHandler.BuildBasedDataSin(XTrain)
    XBassedVal = BaseHandler.BuildBasedDataSin(XVal)
    print XBassedVal
    W, Lambda = LASSO.get_LASSO(XBassedTrain, np.concatenate(YTrain, axis=0),
                                XBassedVal, np.concatenate(YVal, axis=0))
    print W
    print Lambda
