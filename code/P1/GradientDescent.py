def GradientDescent(Function,Gradient,StartingPoint,StepSize,Error,*args):
    x_curr=StartingPoint
    x=[x_curr]
    while True:
        x_prev=x_curr
        x_curr=x_prev-StepSize*Gradient(x_prev,*args)
        x.append(x_curr)
        if abs(Function(x_curr,*args)-Function(x_prev,*args))<Error:
            #print (Function(x_curr, *args))
            break
        if len(x)>10000:
            print "not converging"
            break
    return x,Function(x_curr,*args)

