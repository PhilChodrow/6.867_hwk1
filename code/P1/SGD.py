import random
import numpy as np
import math
import Functions

def SGD(Function,GradientPoint,theta0,tau0,Kappa,Error,Cheating,CeatingVal,X,Y, max_iters = 10000):
    theta_curr = theta0
    theta = [theta_curr]
    t=0
    NumOfElements=len(X)
    while True:
        t = t + 1
        theta_prev=theta_curr
        StepSize=(tau0+t)**(-Kappa)
        SampledPoint=random.randint(0, NumOfElements-1)
        CurrentX=X[SampledPoint]
        CurrentY=Y[SampledPoint]
        theta_curr=theta_prev-StepSize*GradientPoint(theta_prev,CurrentX,CurrentY)
        theta.append(theta_curr)
        BaseVal = Function(theta_prev,X,Y)
        if Cheating:
            BaseVal = CeatingVal
        if abs(Function(theta_curr,X,Y)-BaseVal)<Error:
            #print J(theta_curr, X, Y)
            break
        if t>max_iters:
            print "doesn't converge"
            break
    return theta,Function(theta_curr,X,Y)
