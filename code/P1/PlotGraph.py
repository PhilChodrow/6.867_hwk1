import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from matplotlib import cm


def PlotGraph3D(title, Function, Path,*args):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    # x = y = np.arange(-70, 80, 0.1)
    y = np.arange(-100, 150, 0.1)
    x = np.arange(-80, 150, 0.1)
    X, Y = np.meshgrid(x, y)
    xPath = []
    yPath = []
    zPath = []

    for i in range(0, len(Path)):
        xPath.append(Path[i][0])
        yPath.append(Path[i][1])
        zPath.append(Function([xPath[i], yPath[i]], *args))

    ax.plot(xPath, yPath, zPath, color="blue", linewidth=2.5)
    ax.plot(xPath[0::10], yPath[0::10], zPath[0::10], 'o')
    zs = np.array([Function([x, y], *args) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    # ax.pcolor(X, Y, Z, cmap=cm.winter, alpha=0.2)
    ax.contour(X, Y, Z, 15)
    ax.set_title(title)
    fig.savefig(title+'.png')
