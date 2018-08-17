import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy import integrate


def sigmoid(x, alpha, mu):
    """
    Returns a logistic function's output
    :param x: input, float
    :param alpha: float, param, negative alpha changes the orientation of the curve.
    :param mu: where the split occurs
    :return: float
    """
    return 1 / (1 + np.exp(-alpha * (x - mu)))


def rbf(x, alpha, mu):
    """
    returns the output of a rbf function
    :param x: input, float
    :param alpha: float, param, negative alpha changes the orientation of the curve.
    :param mu: where the split occurs
    :return: float
    """
    return np.exp(-alpha * (x - mu)**2)



alpha = 8
x = np.linspace(-1, 2.5, 2000)
r = 1
uj = .2
#plt.plot(x, [sigmoid(i, alpha, 1) for i in x], label="Sigmoid Function")
#plt.plot(x, [sigmoid(i, alpha, 0) * sigmoid(i, alpha, r) * (i - uj) for i in x], label="orig")
#plt.plot(x, [(uj*(r - uj)*sigmoid(i, alpha, 0) - uj*(r - uj)*sigmoid(i, alpha, r))*0 + i - uj for i in x], label="approx")
#plt.plot(r, uj*(r - uj)*sigmoid(r, alpha, 0)  -uj*(r - uj)*sigmoid(r, alpha, r) + r - uj, 'bo')
#plt.plot(0, uj*(r - uj)*sigmoid(0, alpha, 0)  -uj*(r - uj)*sigmoid(0, alpha, r) -uj, 'bo')
#plt.plot(x, [sigmoid(i, alpha, 0) for i in x], label="10")
#plt.legend()
#plt.ylim((-5, 10))
#plt.title("Sigmoid Function")
#plt.show()


'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.linspace(-1, 1, 40)
Y = np.linspace(-1, 1, 40)
R = np.array([[sigmoid(X[i]**2+Y[j]**2, alpha, 0) for i in range(40)] for j in range(40)])
X, Y = np.meshgrid(X, Y)

Z = R
print(X[0])

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Multivariate Sigmoid Function")
plt.show()












'''
# First sequence element
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [.25 * rbf(i, 4, 0) for i in x], label="rbf")
plt.plot(x, [sigmoid(i, 4, 0) - sigmoid(i, 4, 0)**2 for i in x])
plt.legend()
plt.show()

# Second sequence element ??? Maybe not, if possible it would be nice to get pure rbfs...
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [.125 * (1.24*rbf(i, 4, 0) + sigmoid(i, 4, -.5) - sigmoid(i, 4, .5)) for i in x], label="squence 2")
plt.plot(x, [sigmoid(i, 4, 0) - sigmoid(i, 4, 0)**2 for i in x], label="true")
plt.legend()
plt.show()

# Second sequence element
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [.25 * (rbf(i, 4, 0) + .05*rbf(i, 4, -1) + .05*rbf(i, 4, 1)) for i in x], label="squence 2")
plt.plot(x, [sigmoid(i, 4, 0) - sigmoid(i, 4, 0)**2 for i in x], label="true")
plt.legend()
plt.show()

# Third sequence element
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [.25 * (.955*rbf(i, 4, 0) + .03*rbf(i, 4, -1) + .03*rbf(i, 4, 1) +
                    .06*rbf(i, 4, -.5) + .06*rbf(i, 4, .5)) for i in x], label="squence 2")
plt.plot(x, [sigmoid(i, 4, 0) - sigmoid(i, 4, 0)**2 for i in x], label="true")
plt.legend()
plt.show()


# Approximating the derivative of the rbfs
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [-1.5*sigmoid(i, -3, 0) + 1.5*sigmoid(i, 3, 0) for i in x], label="squence 1")
plt.plot(x, [2 * 3 * rbf(i, 3, 0) * (i - 0) for i in x], label="true")
plt.legend()
plt.show()

# rbfs 2
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [-2.93*sigmoid(i, -3, 0) + 2.93*sigmoid(i, 3, 0)
             - 2.93*sigmoid(i, 3, -1) + 2.93*sigmoid(i, -3, 1) for i in x], label="squence 2")
plt.plot(x, [2 * 3 * rbf(i, 3, 0) * (i - 0) for i in x], label="true")
plt.legend()
plt.show()


# rbfs 3
x = np.linspace(-2.5, 2.5, 200)
plt.plot(x, [-3*sigmoid(i, -3, 0) + 3*sigmoid(i, 3, 0)
             - 2.5*sigmoid(i, 3, -1) + 2.5*sigmoid(i, -3, 1)
             - 2.51*sigmoid(i, -3, -.5) + 2.51*sigmoid(i, 3, .5)
             + .5*sigmoid(i, 3, -1.5) - .5*sigmoid(i, -3, 1.5) for i in x], label="squence 3")
plt.plot(x, [2 * 3 * rbf(i, 3, 0) * (i - 0) for i in x], label="true")
plt.legend()
plt.show()


x = np.linspace(-3, 5, 200)
plt.plot(x, [i / (1-np.exp(i)) for i in x], label="orig")
plt.plot(x, [i / (1-np.exp(i)) * (sigmoid(i, 3, 0) - sigmoid(i, 3, 0)**2) for i in x], label="multiplied")
plt.plot(x, [-(.25 * (.9*rbf(i, 4, 0) + .03*rbf(i, 4, -1) + .03*rbf(i, 4, 1) +
                    .4*rbf(i, 4, -.5) + .06*rbf(i, 4, .5))) for i in x], label="approximation")
plt.legend()
#plt.ylim((-5, 10))
plt.show()



def f(x):
    return np.array([np.sin(x[0]) + x[1], np.exp(-x[3]), x[2]+x[3], x[0]**2])


def obj_func(w, t):
    n = 5
    return integrate.quad(lambda z : (f(z) - L(w, z))**2, [(-np.inf, np.inf) for i in range(n)], )
    

'''