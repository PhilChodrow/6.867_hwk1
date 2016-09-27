import numpy as np
def FiniteDifferenceApproximation(x,DifferenceStep,Function,*args):
    grad=[]
    steps=np.eye(len(x))*DifferenceStep
    for i in range(0,len(x)):
        grad.append((Function(x+steps[i,:],*args)-Function(x,*args))/DifferenceStep)
    return grad
