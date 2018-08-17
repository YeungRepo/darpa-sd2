import numpy as np
from scipy import optimize, integrate
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)


def logistic(x, center, alpha):
    """
    :param x: float, function input
    :param center: float, parameter
    :param alpha: float, parameter
    :return: 
    """
    #if center > 1e2:
        #print(center, x)
    return 1./(1 + np.exp(-alpha * (x - center)))


def mv_logistic(x, mu, alpha, l, n):  # n==len(x) must be true!
    """
    Product of logistic functions
    :param x: array, function input
    :param mu: array, function parameter
    :param alpha: float, parameter
    :param l: int, index of a matrix
    :param n: len(x)
    :return: 
    """
    return np.prod([logistic(x[j], mu[l, j], alpha) for j in range(n)])


def Lambda_T(x, mu, alpha, n, N_L):  # n==len(x) must be true!
    """
    Vector of product of logistic functions
    :param x: 
    :param mu: 
    :param alpha: 
    :param n: 
    :param N_L: 
    :return: 
    """
    return np.array([mv_logistic(x, mu, alpha, l, n) for l in range(N_L)])


def mv_log_prime(x, mu, alpha, l, n, func):
    """
    The derivative of the lth logistic product.
    :param x: 
    :param mu: 
    :param alpha: 
    :param l: 
    :param n: 
    :param func: a function from R^n to R^n
    :return: 
    """
    return mv_logistic(x, mu, alpha, l, n) * alpha * sum([(1 - logistic(x[i], mu[l, i], alpha)) * func(x)[i] for i in range(n)])


def oss1(x):
    """
    System's first state evolution rule (just an example)
    :param x: 
    :return: 
    """
    return x[1]


def oss2(x):
    """
    System's second state evolution rule (just an example)
    :param x: 
    :return: 
    """
    return -x[0] + 0.2 * (1 - x[0]**2) * x[1]


def homo_quad1(x):
    return x[0]**2 + x[0]*x[1]


def homo_quad2(x):
    return .5*x[1]**2 + x[0]*x[1]


def lienard1(x):
    return -x[1] * (1+x[0]) + 0.2*x[0] + (0.2+1)*x[0]**2


def lienard2(x):
    return x[0] * (1 + x[0])


def f(x):
    """
    The state evolution rule for the entire system (just an example)
    :param x: 
    :return: 
    """
    return np.array([oss1(x), oss2(x)])


def g(x, mu, alpha, n, func, N_L):
    """
    The derivative of the lifted state.
    :param x:  state of dimension n 
    :param mu: centers of dimension n_L by n  
    :param alpha: steepness coefficient 
    :param n: dimension of state
    :param func: 
    :param N_L: 
    :return: 
    """
    # First we make each of the Lambda_T entries and the f entries
    L_T = [mv_log_prime(x, mu, alpha, l, n, func) for l in range(N_L)] #mv_log_prime is the derivative of the  multivarite logistic function 
    f_x = func(x)
    return np.hstack([0, f_x, L_T])


def primary_objective_func(points, mu_vec, w_vec, alpha, func, N_L, n):
    """
    This function optimizes centers and weights to approximate func (f(x) the vector field).
    :param points: 
    :param mu_vec: 
    :param w_vec: 
    :param alpha: 
    :param func: 
    :param N_L: 
    :param n: 
    :return: 
    """
    # First we reorganize mu_vec and w_vec into matrices
    dim_w = n + N_L + 1
    mu = mu_vec.reshape(N_L, n)  # mu is a non-square matrix of centers, #rows:  logistic functions, # columns: dim of x
    w = w_vec.reshape(n, dim_w)  # w is the Koopman operator's rows corresponding to f.
    basis_func = lambda x: np.hstack([np.array([1]), x, Lambda_T(x, mu, alpha, n, N_L)])
    return sum([np.linalg.norm(np.multiply((basis_func(x)).dot(w.T) - func(x), np.array([1, 1]))) for x in points])


def lsq_p_objective_func(points, mu_vec, w_vec, alpha, func, N_L, n):
    """
    This function optimizes centers and weights to approximate func (f(x) the vector field).
    It does so for the least squares routine
    :param points: 
    :param mu_vec: 
    :param w_vec: 
    :param alpha: 
    :param func: 
    :param N_L: 
    :param n: 
    :return: 
    """
    # First we reorganize mu_vec and w_vec into matrices
    dim_w = n + N_L + 1
    mu = mu_vec.reshape(N_L, n)  # mu is a non-square matrix of centers, #rows:  logistic functions, # columns: dim of x
    w = w_vec.reshape(n, N_L)  # w is the Koopman operator's rows corresponding to f.
    basis_func = lambda x: np.hstack([Lambda_T(x, mu, alpha, n, N_L)])
    bf_2 = lambda x: np.hstack([np.ones(1), x])
    weights = np.array([[0, 0, 1.], [.2, -.2*7, 1]])
    # The choice of weights below is specific to the van der pol oscillator
    # As is the choice of the dimension of f and w.
    return sum([np.linalg.norm(np.multiply((basis_func(x)).dot(w.T) - func(x), np.array([1, 1]))) for x in points])


def get_Mu_koop_for_f(points, N_L, n, alpha, func, initer):
    """
    Calculates the centers for the koopman operator and weights for approximating f(x).
    :param points: 
    :param N_L: 
    :param n: 
    :param alpha: 
    :param func: 
    :param initer: 
    :return: 
    """
    dim_w = n + N_L + 1
    grid_max = np.max(points)
    #print("grid_max", grid_max)
    # Inpt is a flattened vector of the matrix of decision variables, the upper LH block is the matrix of centers
    # (n by nL) and the lower RH block will be th Koopman weights

    def x1gtx2(inpt, nL_i, nL_j):
        # N_L and n are the # of logistic functions and states respectively.
        # They are defined as global variables, since x1gtx2 is invoked as a constraint function under calc_Mu_K
        all_centers = inpt[0:N_L * n].reshape(N_L, n)
        # check that all entries in the nL_i column are greater than nl_j column
        diff_vec_bool = np.float(np.all(all_centers[nL_i, :] - all_centers[nL_j, :] >= 0)) - 0.5
        return diff_vec_bool

    def centers_in_grid(inpt, nL_i):
        # N_L and n are the # of logistic functions and states respectively.
        # They are defined as global variables, since x1gtx2 is invoked as a constraint function under calc_Mu_K
        mu = inpt[0:N_L * n].reshape(N_L, n)
        # check that all entries in the nL_i column are greater than nl_j column
        ingrid_const = 2 * grid_max - np.linalg.norm(mu[nL_i], ord=np.inf)
        return ingrid_const

    def call(mu_vec):
        if np.linalg.norm(mu_vec[0:N_L*n], ord=np.inf) > 1e6:
            raise OverflowError

    opt_results = optimize.minimize(lambda inpt: primary_objective_func(points, inpt[0:N_L * n], inpt[N_L * n:], alpha, func, N_L, n), x0=initer, method="COBYLA",
                                    constraints=[{'type': 'ineq', 'fun': x1gtx2, 'args': [i, i + 1]} for i in range(0, N_L-1)] + [{'type': 'ineq', 'fun': centers_in_grid, 'args': [i]} for i in range(0, N_L)]
                                    ,  options={"maxiter": 50, "disp": False})#, "ftol":1e-6}, callback=call)

    Mu = opt_results.x[0:N_L * n].reshape(N_L, n)
    K = opt_results.x[N_L * n:].reshape(n, dim_w)
    #print("Result of the optimization routine: {0}".format(str(opt_results.success)))
    return Mu, K


def lsq_Mu_Koop_for_f(points, N_L, n, alpha, func, initer):
    """
    Calculates the centers for the koopman operator and weights for approximating f(x).
    :param points: 
    :param N_L: 
    :param n: 
    :param alpha: 
    :param func: 
    :param initer: 
    :return: 
    """
    dim_w = n + N_L + 1
    grid_max = np.max(points)
    #print("grid_max", grid_max)
    # Inpt is a flattened vector of the matrix of decision variables, the upper LH block is the matrix of centers
    # (n by nL) and the lower RH block will be th Koopman weights

    def x1gtx2(inpt, nL_i, nL_j):
        # N_L and n are the # of logistic functions and states respectively.
        # They are defined as global variables, since x1gtx2 is invoked as a constraint function under calc_Mu_K
        all_centers = inpt[0:N_L * n].reshape(N_L, n)
        # check that all entries in the nL_i column are greater than nl_j column
        diff_vec_bool = np.float(np.all(all_centers[nL_i, :] - all_centers[nL_j, :] >= 0)) - 0.5
        return diff_vec_bool

    def centers_in_grid(inpt, nL_i):
        # N_L and n are the # of logistic functions and states respectively.
        # They are defined as global variables, since x1gtx2 is invoked as a constraint function under calc_Mu_K
        mu = inpt[0:N_L * n].reshape(N_L, n)
        # check that all entries in the nL_i column are greater than nl_j column
        ingrid_const = 2 * grid_max - np.linalg.norm(mu[nL_i], ord=np.inf)
        return ingrid_const

    def call(mu_vec):
        if np.linalg.norm(mu_vec[0:N_L*n], ord=np.inf) > 1e6:
            raise OverflowError

    opt_results = optimize.minimize(lambda inpt: lsq_p_objective_func(points, inpt[0:N_L * n], inpt[N_L * n:], alpha, func, N_L, n), x0=initer, method="COBYLA",
                                    constraints=[{'type': 'ineq', 'fun': x1gtx2, 'args': [i, i + 1]} for i in range(0, N_L-1)] + [{'type': 'ineq', 'fun': centers_in_grid, 'args': [i]} for i in range(0, N_L)]
                                    ,  options={"maxiter": 50, "disp": False})#, "ftol":1e-6}, callback=call)

    Mu = opt_results.x[0:N_L * n].reshape(N_L, n)
    K = opt_results.x[N_L * n:].reshape(n, N_L)
    # Now we improve our approximation with some least squares magic...
    #print(points, np.ones(len(points)))
    A = np.vstack([np.ones(len(points)), np.array(points).T]).T
    W = np.zeros([n, n + 1])
    for row in range(n):
        b = np.array([func(x)[row] - K[row].dot(np.array([mv_logistic(x, Mu, alpha, l, n) for l in range(N_L)])) for x in points])
        W[row, :] = np.linalg.lstsq(A, b)[0]
    K = np.hstack([W, K])
    return Mu, K


def secondary_objective_function(points, mu_vec, w_vec, alpha, func, N_L, n):
    """
    This function optimizes centers and weights to approximate func (f(x) the vector field).
    :param points: 
    :param mu_vec: 
    :param w_vec: 
    :param alpha: 
    :param func: 
    :param N_L: 
    :param n: 
    :return: 
    """
    # First we reorganize mu_vec and w_vec into matrices
    dim_w = n + N_L + 1
    mu = mu_vec.reshape(N_L, n)  # mu is a non-square matrix of centers, #rows:  logistic functions, # columns: dim of x
    w = np.hstack([np.zeros([N_L, n + 1]), w_vec.reshape(N_L, N_L)])  # w is the Koopman operator's rows corresponding to f.
    # print(dim_w-1, np.shape(Lambda_T(points[0], mu, alpha, n, N_L))[0]+ n)
    basis_func = lambda x: np.hstack([np.array([1]), x, Lambda_T(x, mu, alpha, n, N_L)])
    return sum([np.linalg.norm((basis_func(x)).dot(w.T) -
                               np.array([mv_log_prime(x, mu, alpha, l, n, func) for l in range(N_L)])) for x in points])

def lsq_rest_of_K(points, N_L, n, alpha, func, mu):
    """
    Calculates the bottom N_L rows of the Koopman operator assuming that the centers have already been chosen.
    Does so with a least squares method.
    :param points: 
    :param N_L: 
    :param n: 
    :param alpha: 
    :param func: 
    :param mu: a matrix 
    :param initer: 
    :return: 
    """
    len_points = len(points)
    W = np.zeros([N_L, N_L])
    count = n + 1
    A = np.array([[mv_logistic(points[i], mu, alpha, j, n) for j in range(N_L)] for i in range(len_points)])
    for row in range(N_L):
        count += 1
        print("Working on row: ", count, " of the Koopman operator.")
        b = np.array([mv_log_prime(points[i], mu, alpha, row, n, func) for i in range(len_points)])
        print(A.shape, b.shape)
        W[row, :] = np.linalg.lstsq(A, b)[0]
    return np.hstack([np.zeros([N_L, n + 1]), W])


def calc_rest_of_K(points, N_L, n, alpha, func, mu_vec, initer):
    """
    Calculates the bottom N_L rows of the Koopman operator assuming that the centers have already been chosen.
    :param points: 
    :param N_L: 
    :param n: 
    :param alpha: 
    :param func: 
    :param mu_vec: 
    :param initer: 
    :return: 
    """
    # Inpt is a flattened vector of the matrix of decision variables, the upper LH block is the matrix of centers
    # (n by nL) and the lower RH block will be th Koopman weights
    opt_results = optimize.minimize(lambda inpt: secondary_objective_function(points, mu_vec, inpt, alpha, func, N_L, n)
                                    , x0=initer, options={"maxiter": 250, "disp": True, "ftol":1e-6})
    K_bot = opt_results.x.reshape(N_L, N_L)
    print("Result of the optimization routine: {0}".format(str(opt_results.success)))
    return np.hstack([np.zeros([N_L, n + 1]), K_bot])


# Now we test the code above and try to learn then simulate the behavior of our operator.
# We use nonlinear system class to do so.

class NonlinearSystem:
    """
    Class for a non-linear system,
    Methods: 
    """
    def __init__(self, func, n, alpha, starting_vals, N_L, mu, k):
        """
        :param func: function discribing system evolution, R^n to R^n
        element's derivative.
        :param alpha: 
        :param starting_vals: array discribing the initial condition of the system.
        """
        self.n = n
        self.x0 = starting_vals
        self.f = func
        self.m = N_L
        self.alpha = alpha
        self.koop = k
        self.mu = mu
        self.ksim = None
        self.nsim = None
        self.npm = self.m + self.n

    def lift_state(self):
        """
        :return: the lifted state.
        """
        return np.hstack([1, self.x0, Lambda_T(self.x0, self.mu, self.alpha, self.n, self.m)])


    def simulate_koop(self, times):
        """
        runs a simulation for the koopman operator state
        :param times: array of times to show the state behavior at.
        :return: an array of the behavior of the system...
        """
        ex0 = self.lift_state()
        self.ksim = integrate.odeint(lambda x, t: self.koop.dot(x), ex0, times)

    def simulate_reg(self, times):
        """
        runs a simulation of the normal system to use for comparison.
        :param times: 
        :return: 
        """
        self.nsim = integrate.odeint(lambda x, t: self.f(x), self.x0, times)

    def plot(self, koopman=True, formula=True, indx=0):
        """
        Plots the most recent simulation results
        :return: 
        """
        if koopman and formula:
            # We plot the two simulations side by side
            plt.plot(self.ksim.T[indx+1], label="Evolved with the Koopman Operator")
            plt.plot(self.nsim.T[indx], label="Evolved by the Dynamics Formula")
            plt.legend()
            plt.title("Comparison of Koopman Evolution and Formula Evolution")
            plt.ylim(-4, 4)
            plt.show()
        elif koopman:
            plt.plot(self.ksim.T[indx+1], label="Evolved with the Koopman Operator")
            plt.legend()
            plt.title("State Evolution")
            plt.show()
        elif formula:
            plt.plot(self.nsim.T[indx], label="Evolved by the Dynamics Formula")
            plt.legend()
            plt.title("State Evolution")
            plt.show()
        else:
            print("Neither koopman nor formula were chosen to be True.")


# We make a function for printing 3d surface plots...

def three_D_surface_plot(func, dist_from_origin, title):
    """
    Plots a 3d surface plot
    :param func: a function from R^2 to R.
    :param dist_from_origin: float
    :param title: string, title of the plot
    :return: 
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-dist_from_origin, dist_from_origin, 0.2)
    Y = np.arange(-dist_from_origin, dist_from_origin, 0.2)
    X, Y = np.meshgrid(X, Y)
    #print("X", X.shape, X[0, 0])
    #print("Y", Y.shape, Y[0, 0])
    #R = func(np.array([X, Y]))
    Z = np.array([[func(np.array([X[i, j], Y[i, j]])) for j in range(len(X[0]))] for i in range(len(X[0]))])
    #print("Z", Z.shape)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-20.01, 20.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()


space = 0.2
size = 3
# Test this



# We now set the parameters for the simulation and learning problem.

point_mesh = np.mgrid[-size:size:space,-size:size:space]
all_x = np.ndarray.flatten(point_mesh[0])
all_y = np.ndarray.flatten(point_mesh[1])
points = [[all_x[i], all_y[i]] for i in range(0,len(all_x))]


# points = [[np.random.randn()*2.2, np.random.randn()*2.2] for i in range(40)]
# points = np.loadtxt("points attempt 2")
# print(points)

N_L = 9  # This is the number of logistic functions
n = 2  # This is the number of centers
alpha = 1.25
x0 = np.array([.5, -.5])
basis_func = lambda x: np.hstack([np.array([1]), x, Lambda_T(x, Mu, alpha, n, N_L)])
lowest_error = np.inf
count = 0
len_pts = len(points)
print("There are ", len_pts, " points.")
######################################################################################
for initer_iter in range(4):
    try:
        initer = np.random.randn(N_L*n + (N_L+n+1)**2) * np.sqrt(size)  #np.hstack([np.loadtxt("alpha, initer, x0 5"), np.random.randn(N_L*n + (N_L+n+1)**2) * .04])
        #print("Initial point of the optimization problem: {0}".format(str(initer)))
        Mu, K_top = lsq_Mu_Koop_for_f(points, N_L, n, alpha, f, initer[0:N_L*n + N_L * n])
        # Just looking at the second element is specific to the van der pol oscillator
        #####################################################################################
        # This code is specific to the van der pol oscillator
        K_top[0,:] = np.hstack([np.zeros(1), np.zeros(1), np.ones(1), np.zeros(N_L)])
        #####################################################################################
        error = np.linalg.norm([f(points[i]) - K_top.dot(basis_func(points[i])) for i in range(len_pts)], ord=1) / float(len_pts)
        if error < lowest_error:
            print("Lowest average error at each point so far: ", error)
            lowest_error = error
            lowest_initer = initer
            best_K_top, best_Mu = K_top, Mu
    except OverflowError:
        count += 1
        print("divergence {0}".format(str(count)))
        continue

print("The centers ", best_Mu)
K_top = best_K_top
Mu = best_Mu
f_error = error
print("The approximation of f:", f_error)


three_D_surface_plot(oss1, size, "f_1(x), actual")

three_D_surface_plot(lambda x: K_top[0,:].dot(basis_func(x)), size,
                     "f_1 N_L={0}, alpha={1}, g-spacing={2}, g-size={3}".format(str(N_L), str(alpha), str(space), str(size)))
three_D_surface_plot(oss2, size, "f_2(x), actual")
three_D_surface_plot(lambda x: K_top[1,:].dot(basis_func(x)), size,
                     "f_2 N_L={0}, alpha={1}, g-spacing={2}, g-size={3}".format(str(N_L), str(alpha), str(space), str(size)))
'''
plt.plot([f([i/5., i/5.]) - K_top.dot(basis_func([i/5., i/5.])) for i in range(10)], label="error in approximating f")
plt.legend()
plt.title("Error in approximating f on the trajectory y=x")
plt.show()
next_points = [Mu[row, :] for row in range(N_L)] + [Mu[row, :] + np.random.randn() for row in range(N_L)] \
              + [Mu[row, :] + np.random.rand()*0.1 for row in range(N_L)]'''
#initer = np.random.randn(N_L*n + (N_L+n+1)**2)
#K_bot = calc_rest_of_K(points, N_L, n, alpha, f, Mu.reshape(1, n*N_L), initer[0:N_L * N_L])
K_bot = lsq_rest_of_K(points, N_L, n, alpha, f, Mu)
print(np.array([mv_log_prime(points[0], Mu, alpha, l, n, f) for l in range(N_L)]).shape, K_bot.dot(basis_func(points[0])).shape)
error = np.linalg.norm([np.array([mv_log_prime(points[i], Mu, alpha, l, n, f) for l in range(N_L)]) - K_bot.dot(basis_func(points[i])) for i in range(len_pts)], ord=1) / float(len_pts)

K = np.vstack([np.zeros(n+N_L+1), K_top, K_bot])
######################################################################################

atpt_num = str(11)  # Change this value to stop saving over yourself. (attempt number)
title_mu = "mu attempt {0}".format(atpt_num)
title_k = "k attempt {0}".format(atpt_num)
title_points = "points attempt {0}".format(atpt_num)
title_hyper = "alpha, initer, x0 {0}".format(atpt_num)
# Save and store the results for Mu and K
np.savetxt(title_mu, Mu)
np.savetxt(title_k, K)
np.savetxt(title_points, points)
np.savetxt(title_hyper, lowest_initer)

example = NonlinearSystem(f, n, alpha, x0, N_L, Mu, K)
print("eigenvalues", np.linalg.eig(K))

# Run the simulation to plot the result of this choice of Mu and K
times = np.linspace(0, 80, 240)
example.simulate_koop(times)
example.simulate_reg(times)
example.plot(indx=0)
example.plot(indx=1)

print(example.ksim[:, 1:3] - example.nsim)
print("The approximation of f:", f_error)
print("Error of the derivative terms", error)
# print(example.nsim)
