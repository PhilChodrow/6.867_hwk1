import random
import numpy as np
import math


def SGD(theta0,tau0,Kappa,Error,X,Y):
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
        theta_curr=theta_prev-StepSize*GradientPointJ(theta_prev,CurrentX,CurrentY)
        theta.append(theta_curr)

        if abs(J(theta_curr,X,Y)-J(theta_prev,X,Y))<Error:
            #print J(theta_curr, X, Y)
            break
        if t>math.exp(20):
            print "doesn't converge"
            break
    return theta,J(theta_curr,X,Y)

def J(theta_t,X,Y):
    val=0
    for i in range(0,len(X)):
        val=val+((np.dot(np.array(X[i]).T,np.array(theta_t))-Y[i])**2)
    return val

def GradientPointJ(theta_t,x,y):
    return 2*(np.dot(np.array(x).T,np.array(theta_t))-y)*np.array(x)

#def GradientJ():
