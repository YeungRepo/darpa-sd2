# Error terms for figures in the paper.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


x = np.linspace(-10, 10, 360)
y = np.linspace(-1, 1, 360)


def m1_error(e, a):  # Max possible error given by eq 14 in the paper as of monday Aug 14 2017
    return abs(a / (1 + 2 * np.exp(a*e/2) + np.exp(a*e)))


eps, alph = np.meshgrid(x, y)
Z = m1_error(eps, alph).T

fig, ax = plt.subplots()
image1 = ax.pcolor(eps, alph, Z, cmap=cm.seismic)
image2 = ax.contour(eps, alph, Z, cmap=cm.RdBu)
color_bar1 = fig.colorbar(image1, label="Error")
color_bar2 = fig.colorbar(image2)
plt.ylabel("epsilon")
plt.xlabel("alpha")
plt.title("Maximum Error as a Function of eps and alph")
plt.show()

########################################################################################################################
# Next error term.
########################################################################################################################

def r1_error(e, a):  # Error when x=eps on
    return abs(a / (1 + 2 * np.exp(a*e/2) + np.exp(a*e)))


eps, alph = np.meshgrid(x, y)
Z = error(eps, alph).T

fig, ax = plt.subplots()
image = ax.pcolor(eps, alph, Z, cmap=cm.seismic)
color_bar = fig.colorbar(image, label="Error")
plt.ylabel("epsilon")
plt.xlabel("alpha")
plt.title("Maximum Error as a Function of eps and alph")
plt.show()
