from sklearn import linear_model
import numpy as np
import numpy.linalg as linalg


def LASSO(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, Lambda):
    clf = linear_model.Lasso(alpha=Lambda)
    # DataSet should look like:[[0, 0], [1, 1], [2, 2]], [0, 1, 2]
    clf.fit(DataSetXTrain, DataSetYTrain)
    Err= clf.predict(DataSetXVal)-np.array(DataSetYVal).T
    RSQ = np.dot(np.array(Err), np.array(Err).T)
    return clf.intercept_,clf.coef_, RSQ


def Ridge(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, Lambda):
    DataSetXTrain = np.concatenate((np.ones((len(DataSetXTrain), 1)), np.array(DataSetXTrain)), axis=1)
    DataSetXVal = np.concatenate((np.ones((len(DataSetXVal), 1)), np.array(DataSetXVal)), axis=1)
    Phi = np.array(DataSetXTrain)
    W = np.dot(np.dot(linalg.inv(Lambda*np.eye(len(Phi[0]))+np.dot(Phi.T, Phi)), Phi.T), DataSetYTrain)
    Err=np.dot( DataSetXVal,W) - DataSetYVal
    RSQ = np.dot(np.array(Err).T, np.array(Err))
    return W[0],W[1:], RSQ

