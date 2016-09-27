import math
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

def BuildBasedData(x,M,type):
    if type == "p":
        DataSet=[]
        for i in range(0,len(x)):
            DataPoint=[]
            for j in range(0,M):
                 DataPoint.append(x[i]**j)
            DataSet.append(DataPoint)
        return DataSet
    if type =="c":
        DataSet = []
        for i in range(0, len(x)):
            DataPoint = [x[i]]
            for j in range(1, M):
                DataPoint.append(math.cos(math.pi * x[i] * j))
            DataSet.append(DataPoint)
        return DataSet
    if type == "s":
        DataSet = []
        for i in range(0, len(x)):
            DataPoint = [x[i][0]]
            for j in range(1, 12):
                DataPoint.append(math.sin(0.4 * math.pi * x[i] * j))
            DataSet.append(DataPoint)
        return DataSet