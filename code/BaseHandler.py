import math
import numpy
def BuildBasedDataPoli(x,M):
    DataSet=[]
    for i in range(0,len(x)):
        DataPoint=[]
        for j in range(0,M):
             DataPoint.append(x[i]**j)
        DataSet.append(DataPoint)
    return DataSet

def BuildBasedDataSin(x):
    DataSet=[]
    for i in range(0,len(x)):
        DataPoint=[x[i][0]]
        for j in range(1,12):
             DataPoint.append(math.sin(0.4*math.pi*x[i]*j))
        DataSet.append(DataPoint)
    return DataSet

def BuildBasedDataCos(x,M):
    DataSet=[]
    for i in range(0,len(x)):
        DataPoint=[x[i][0]]
        for j in range(1,M):
             DataPoint.append(math.cos(math.pi*x[i]*j))
        DataSet.append(DataPoint)
    return DataSet

def BuildBasedData(x,M,FuncrType):
    if FuncrType == "p":
        DataSet=[]
        for i in range(0,len(x)):
            DataPoint=[]
            for j in range(0,M):
                DataPoint.append(x[i][0]**j)
            DataSet.append(DataPoint)
        return DataSet
    if FuncrType =="c":
        DataSet = []
        for i in range(0, len(x)):
            DataPoint = [x[i]]
            for j in range(1, M+1):
                DataPoint.append(math.cos(math.pi * x[i] * j))
            DataSet.append(DataPoint)
        return DataSet
    if FuncrType == "s":
        DataSet = []
        for i in range(0, len(x)):
            if (type(x[i]) is numpy.float64):
                DataPoint = [x[i]]
            else:
                DataPoint = [x[i][0]]
            for j in range(1, M):
                DataPoint.append(math.sin(0.4 * math.pi * x[i] * j))
            DataSet.append(DataPoint)
        return DataSet