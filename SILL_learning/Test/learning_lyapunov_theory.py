import numpy as np
import control

# The purpose of this is to create a codebase for pedagogical purposes
# I want to have ready systems for learning about and testing control theory concepts.

# global variables for defining a 2 state nonlinear system
# 4.1
a = 1


def h(x):
    return np.sin(x)


def d_x1(x):
    """returns the derivative of x1"""
    return x[1]


def d_x2(x):
    """returns the derivative of x2"""
    return -h(x[0]) - a * x[1]


# With these update rules we have a system developed.
# Now we determine its stability by coming up with a lyapunov function.


# 4.2 System representation for a neural network
# important global variables, and functions
len_x = 2  # The number of state variables
C = [1, .5]  # capacitance at the ith amplifier unit
T = np.array([[1, 2], [2, 4]])  # matrix of signed conductances
p = [.2, .3]  # resistance at the ith amplifier units
V_m = 1  # Asymptote of the sigmoid function
lamb = 6  # parameter, slope of g at x = 0, always positive
I = [1, 4]  # Constant input current
one_over_R = [1/p[i] + sum([np.linalg.norm(T[i, j]) * np.sign(T[i, j]) for j in range(len_x)]) for i in range(len_x)]


def g(x):
    return 2/np.pi * V_m * np.arctan(lamb * np.pi * x / (2 * V_m))


def g_inv(y):
    return 2*V_m / (np.pi * lamb) * np.tan(y*np.pi / (2*V_m))


# The state evolution rule.
def d_xi(x, i):
    """returns the derivative of the ith voltage (state variable) of the neural network"""
    return 1/C[i] * h(x[i])[i] * (sum([T[i, j] * x[j] for j in range(len(x))]) - one_over_R[i] * g_inv(x[i]) + I[i])