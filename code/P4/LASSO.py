from sklearn import linear_model
import numpy as np
def get_LASSO(DataSetXTrain,DataSetYTrain,DataSetXVal,DataSetYVal):
    BestW=[]
    BestRSQ=float('Inf')
    BestLambda=0
    for Lambda in np.arange(0.1, 1, 0.01):
        clf = linear_model.Lasso(alpha = Lambda)
        #DataSet should look like:[[0, 0], [1, 1], [2, 2]], [0, 1, 2]
        clf.fit(DataSetXTrain,DataSetYTrain)
        RSQ= np.dot(np.array(clf.predict(DataSetXVal)-DataSetYVal).T,np.array(clf.predict(DataSetXVal)-DataSetYVal))
        print RSQ
        if RSQ<BestRSQ:
            BestW=[clf.intercept_],clf.coef_
            BestLambda=Lambda
            BestRSQ=RSQ
    return BestW,BestLambda

