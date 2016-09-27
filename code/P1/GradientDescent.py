def GradientDescent(Function,Gradient,StartingPoint,StepSize,Error,Cheating,CeatingVal,max_iters = 10000, *args):
    x_curr=StartingPoint
    x=[x_curr]
    while True:
        x_prev=x_curr
        x_curr=x_prev-StepSize*Gradient(x_prev,*args)
        x.append(x_curr)
        BaseVal=Function(x_prev,*args)
        if Cheating:
            BaseVal=CeatingVal
        if abs(Function(x_curr,*args)-BaseVal)<Error:
            #print (Function(x_curr, *args))
            break   
        if len(x)>max_iters:
            print "not converging"
            break
    return x,Function(x_curr,*args)

