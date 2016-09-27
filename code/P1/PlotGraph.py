import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from matplotlib import cm

def PlotGraph3D(title,Function,Path,*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-10, 50, 0.05)
    X, Y = np.meshgrid(x, y)
    xPath = []
    yPath = []
    zPath = []

    for i in range(0, len(Path)):
        xPath.append(Path[i][0])
        yPath.append(Path[i][1])
        zPath.append(Function([xPath[i], yPath[i]], *args))


    ax.plot(xPath, yPath, zPath,'-b')

    zs = np.array([Function([x, y],*args) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z,cmap=cm.winter, alpha=0.2)
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
    ax.set_title(title)

    #plt.show()