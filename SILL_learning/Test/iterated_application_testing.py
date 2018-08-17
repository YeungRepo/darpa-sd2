import numpy as np
from matplotlib import pyplot as plt


# Define a function to apply and plot another function over and over again
def plot_iters(func, x0, num_iters, state_indx, title):
    """
    Iterativly applies the function (that maps from R^n to R^n) to the initial conditions num_iters times and then plots the evolution of the 
    state in question.
    :param func: a function to be applied (from R^n to R^n), a difference equation
    :param x0: a numpy array of length n of the initial state to be applied into the function
    :param num_iters: how many times to apply the function, int
    :param state_indx: int, which state to plot, 0-indexed
    :param title: str, the name of the plot
    :return: state, a 2d numpy array of the state at each of the num_iters.
    """
    state = np.zeros([len(x0), num_iters])
    # Start filling in state
    state[:, 0] = x0
    for i in range(num_iters - 1):
        x0 = func(x0)  # Set the new initial state
        state[:, i + 1] = x0
    plt.plot(state[state_indx, :])
    plt.xlabel("Number of Iterations")
    plt.ylabel("Value of x{0}".format(str(state_indx)))
    plt.title(title)
    plt.show()
    return state


# ans = plot_iters(np.cos, np.array([0.]), 150, 0, "cos(cos(...(0)...))")
# print(ans)

