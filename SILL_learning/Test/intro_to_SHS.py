# This file is an exploration of Stochastic Hybrid Systems
import numpy as np


Q = {"s1", "s2", "s3"}

def d(q):
    """
    A map from Q to the natural numbers
    :param q: an element of Q
    :return: number, an int between 1 and 3
    """
    if q == "s1":
        return 1
    elif q == "s2":
        return 2
    elif q == "s3":
        return 3
    else:
        return "Did not pass in correct parameter."

def exe(q):
    """
    A map from Q to an open interval of R1, R2 or R3, in this case all are intervals from -1 to 1.
    :param q: an element of Q
    :return: a 1, 2, or 3 row array of open interval boundries (one for each dimension)
    """
    return tuple((-1, 1) for i in range(d(q)))

# This is the hybrid state space for our GSHS
EX = {(q, exe(q)) for q in Q}


def b(q, x):
    """
    a differental equation.
    This is a vector field that moves elements of the output of exe(q), 
    corresponds to drift
    :param q: the mode of the system at the time
    :param x: the state of the system
    :return: 
    """
    return None

def sigma(q, x):
    """
    This returns a d(q) by m matrix, m\in\mathbb{N}, 
    corresponds to diffusion 
    :param q: the mode of the system at the time
    :param x: the state of the system
    :return: 
    """



def init():
    """
    The initial probability measure on (X, B(S)))
    S is the stopping time?
    :return: 
    """


def Lmbda():
    """
    lambda is a transition rate function,
    it takes elements from the closure of the inputs to b
    :return: 
    """


def arr():
    """
    R is a transition measure
    it takes inputs from the closure of 
    :return: 
    """


def wiener_motion(dim, current, tstep):
    """
    Returns the next step in wiener_motion
    :param dim: the dimension of the state
    :param current: the states current values
    :param tstep: time step
    :return: the next position of the state
    """
    perts = [np.random.normal() for i in range(dim)]
    t = np.sqrt(tstep)
    for i in range(dim):
        current[i] += perts[i] * t
    return current



























# This below is taken from the readme file of pysde, downloaded Friday May 26, 2017
# Symbolic Solution
from sympy import *
from sympy .abc import t,x,k,N,m,C
import pysde
import matplotlib.pylab as plt
from matplotlib import rc
from sympy import *
from pysde.sde import *
'''
drift = -2 * x / (1 + t)
diffusion = sqrt(t * (1 - t))
t0 = Symbol('x0')
x0 = Symbol('t0')
sol = pysde.SDE_solver(drift, diffusion, t0, x0)
print(sol)

# Numerical Solution (simulation)
import matplotlib.pylab as plt
from matplotlib import rc
from sympy import *
from pysde.sde import *

""" setup picture info """
plt.figure(figsize=(5,2))
plt.ylim(-0.5, 1.5)
""" Initial data """
x0 = 1.
t0 = 0.
tn = 10.
x, dx = symbols('x dx')
a, b, c, d = 0, -1., 0, 1.  # Change these values for some different equations
drift = a + b * x  # This is the time domain element of the SDE
diffusion = c + d * x  # This is the stochastic element of the SDE.
nt = 200
T = np.linspace(t0, tn, nt+1)
""" Numerical Computation"""
print("x,t,t,n", x0, t0, tn, nt)
X = pysde.sde.Euler(drift, diffusion, x0, t0, tn, nt)
X, Y = pysde.sde.Milstein(drift, diffusion, x0, t0, tn, nt)
"""Make picture"""
plt.plot(T, X, color="blue", linewidth=2.5, linestyle="-", label="Euler")
plt.plot(T, Y, color="red", linewidth=2.5, linestyle="--", label="Milstein")
plt.plot(T, np.exp(-T), color="green", linewidth=2.5, linestyle="--", label=r"$\exp(-t)$")
plt.ylim(X.min()-0.2, X.max()+0.2)
plt.title(r"$d X_t=-dt+d W_t,X_0=1$")
plt.legend()
plt.show()
'''
# End of material from pysde's readme file.


# Ok, lets actually build a SHS!
from scipy import integrate

Q = {"on", "off"}
d = {"on": 1, "off": 1}
# We let X be the real numbers
Init = [np.random.poisson() % 2, np.random.normal()]
drift = {"on": -1 * x, "off": 0.5 * x}  # b
diffusion = {"on": 1 * x, "off": -1.1 * x}  # sigma
lmbda = lambda w: 5000.1 * w
delta_t = 0.1

def mode_trip(q, x0, t_max):
    """
    A function to determine the trajectory of the system in one mode and the stopping time,
    where it switches to another mode.
    :param q: the mode of the system.
    :param x0: the starting condition of the state
    :param t_max: the highest that the time can go in this run
    :return: stop_time (float) and trajectory (array)
    """
    ess = [lmbda(i * delta_t) for i in range(int(t_max/delta_t))]  # Approximating integration
    flag = True
    es = 0
    tot = 0.
    for i in range(len(ess)):
        tot += ess[i]
        if flag and tot > 0.1:
            es = i + 1
            flag = False
        if flag:
            break
    if flag:
        es = np.infty
    print('x,t,t,n', x0, 0, t_max, int(t_max/delta_t))
    trajectory = pysde.sde.Euler(drift[q], diffusion[q], x0, 0, t_max, int(t_max/delta_t))
    ei = 1
    while ei < len(trajectory) and trajectory[ei] < 2:
        ei += 1
    return delta_t * min(es, ei), trajectory

def arre(q, x_t):
    """
    Choses the new inital state (the new mode is obvious as there are only two).
    :param q: the mode that we are switching from (current mode)
    :param x_t: the final conditions in that mode
    :return: the new initial conditions
    """
    if q == "on":
        return 0.2 * x_t
    elif q == "off":
        return 3 * x_t
    else:
        return "q is neither on nor off"

def simulate(q0, end_time):
    """
    Simulates a SHS until the end time and plots its behavior in the end.
    :param q0: does the system start "on" or "off" (chosen randomly)
    :param end_time: how long to run the simulation
    :return: x_stream, the state (x) as it evolves through time.
    """
    time_remaining = end_time
    x_stream = []
    x_t = np.random.normal()  # Choose initial state
    while time_remaining > 0:
        stop_time, trajectory = mode_trip(q0, x_t, time_remaining)  # Determine the stopping time
        # Determine the new initial state
        x_t = arre(q0, trajectory[-1])
        # update how much time is left in the simulation
        time_remaining -= stop_time
        x_stream.append(trajectory)
        if q0 == "on":
            q0 = "off"
        elif q0 == "off":
            q0 = "on"
        else:
            return "q is neither on nor off."
    x_strm = [val for trajec in x_stream for val in trajec]
    T = np.linspace(0, end_time, len(x_strm))
    """Make picture"""
    plt.plot(T, x_strm, color="blue", linewidth=2.5, linestyle="-", label="Euler")
    plt.ylim(min(x_strm) - 0.2, max(x_strm) + 0.2)
    plt.title("My Stochastic Hybrid System")
    plt.legend()
    plt.show()
    return x_strm



# Now we test the system:

print(len(simulate("on", 100)))

# Kinda lame, but kinda cool.