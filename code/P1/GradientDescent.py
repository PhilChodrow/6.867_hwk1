import numpy as np
import math


def GradientDescent(Function, Gradient, StartingPoint, StepSize, Error, Cheating, CeatingVal, max_iters=10000, *args):
    x_curr = StartingPoint
    x = [x_curr]
    grad = [math.sqrt(np.dot(np.array(Gradient(x_curr, *args)), np.array(Gradient(x_curr, *args))))]
    while True:
        x_prev = x_curr
        x_curr = x_prev-StepSize*Gradient(x_prev, *args)
        x.append(x_curr)
        grad.append(math.sqrt(np.dot(np.array(Gradient(x_curr, *args)), np.array(Gradient(x_curr, *args)))))
        BaseVal = Function(x_prev, *args)
        if Cheating:
            BaseVal = CeatingVal
        if abs(Function(x_curr, *args)-BaseVal) < Error:
            break   
        if len(x)>max_iters:
            print "not converging"
            break
    return x, Function(x_curr, *args), grad

