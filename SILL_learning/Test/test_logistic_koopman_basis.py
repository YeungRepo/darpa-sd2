import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy import optimize, integrate
from functools import partial


def logistic(x, alpha):
    return (1 + np.exp(-alpha*x))**-1


def ef(x):
    return np.array([x[0]**2, x[1]**2, x[2] * np.exp(x[0]+x[1])])


class NonlinearSystem:
    """
    Class for a non-linear system,
    Methods: 
    """
    def __init__(self, fs, centers, alpha, starting_vals):
        """
        :param fs: a list of each of the functions, f_i that discribe the derivatives of the ith state 
        element's derivative.
        :param centers: 
        :param alpha: 
        :param starting_vals: list
        """
        self.n = len(fs)
        self.x0 = starting_vals
        self.fs = fs
        self.centers = centers
        self.m = len(self.centers)
        self.alpha = alpha
        self.koop = np.zeros([self.m+self.n, self.m+self.n])
        self.ksim = None
        self.nsim = None
        self.npm = self.m + self.n

    def koop_row(self, func, approx_pts, g, sum_g):
        """
        Returns the weights for one row of the koopman operator.  It chooses weights that will interpolate at the
        center points.
        :param func: the function that we are trying to interpolate, should take in a vector of length n
        :param eps: spacing of approximation points from the centers.
        :return: An array that is n + m in length (the first n entries are zero...)
        """
        #to_min = lambda w, c: sum([(func(w[i]) - sum_g(w[i], w[1000 + 1])) ** 2 for i in range(1000)])
        to_min = lambda w: sum([(func(mu) - sum_g(mu, w))**2 for mu in approx_pts])
        #jacobian = lambda w: np.array([2 * sum([g(mu)[i] * (func(mu) - sum_g(mu, w)) for mu in approx_pts])
        #                               for i in range(1+self.npm)])
        weights = optimize.minimize(to_min, np.zeros(self.npm+1), method="BFGS")#, jac=jacobian)#, hess=hessian)
        print("Optimizaiton Sucessfull ", weights.success)
        return weights.x

    def learn_koop_opt(self):
        """
        Numerically find an approximation to the koopman operator for this system
        :param eps: float, a small value to determine spacing of approximation points
        :return: nothing
        creates a koop value for the object, this is a dim by dim matrix representing an approximated 
        koopman operator for the system.
        """
        approx_pts = [np.array([i, j]) for i in np.linspace(-2,2,10) for j in np.linspace(-4,4,20)]
        en = lambda y: np.linalg.norm(y) ** 2
        g = lambda x: np.array([1] + [x[i] for i in range(self.n)] + \
                               [logistic(en(x - self.centers[i]), self.alpha) for i in range(len(self.centers))])
        sum_g = lambda x, w: g(x).dot(w)
        g_prime = lambda i, x: sum([2 * (x[j] - self.centers[i][j]) * self.fs[j](x) *
                                 (logistic(en(x - self.centers[i]), self.alpha) -
                                  logistic(en(x - self.centers[i]), self.alpha) ** 2) for j in range(self.n)])
        funcs = [lambda x: 0] + self.fs + [partial(g_prime, i) for i in range(self.m)]
        #hessian = lambda w: 2 * np.array([[sum([g(mu)[i] * g(mu)[j] for mu in approx_pts])
        #                                   for j in range(1 + self.npm)] for i in range(1 + self.npm)])
        # Build our koopman operator
        K = np.zeros((self.npm+1, self.npm+1))
        for row in range(self.npm+1):
            K[row] = self.koop_row(funcs[row], approx_pts, g, sum_g)
            print("Koopman Row ", row)
        self.koop = K

    def learn_koop_learn_pts(self, k):
        """
        Learn the koopman operator and learn the points to learn it at simultaniously
        :param k: int, the number of approximating points to use when learning.
        :return: 
        """
        approx_pts = [np.array([i, j]) for i in np.linspace(-2,2,k) for j in np.linspace(-2,2,k)]
        en = lambda y: np.linalg.norm(y) ** 2
        g_prime = lambda i, x: sum([2 * (x[j] - self.centers[i][j]) * self.fs[j](x) *
                                    (logistic(en(x - self.centers[i]), self.alpha) -
                                     logistic(en(x - self.centers[i]), self.alpha) ** 2) for j in range(self.n)])
        funcs = [lambda x: 0] + self.fs + [partial(g_prime, i) for i in range(self.m)]
        # Build our koopman operator
        K = np.zeros((self.npm + 1, self.npm + 1))
        row = 0
        for f in funcs[1:]:
            row += 1
            to_min = lambda inpt: sum([(inpt[0] + sum([inpt[l+1] * x[l] for l in range(self.n)]) +
                                        sum([inpt[i+self.n] * logistic(en(x - inpt[self.m+(i+1)*self.n:self.m+(i+2)*self.n]), self.alpha)
                                             for i in range(self.m)]) - f(x))**2 for x in approx_pts])
            weights = optimize.minimize(to_min, np.random.rand(self.npm + 1 + self.m*self.n),
                                        method="CG")  # , jac=jacobian)#, hess=hessian)
            print("Optimizaiton Sucessfull ", weights.success, "weights", weights.x[0:12])
            K[row] = weights.x[0:self.m+self.n+1]
        self.koop = K

    def learn_koop_lsq(self):
        """
        Numerically find an approximation to the koopman operator for this system
        :param eps: float, a small value to determine spacing of approximation points
        :return: nothing
        creates a koop value for the object, this is a dim by dim matrix representing an approximated 
        koopman operator for the system.
        """
        # Create a list of all the functions that we will need to calculate to get our row choices for the KO
        en = lambda y: np.linalg.norm(y)**2
        sean = lambda i, x: sum([2 * (x[j] - self.centers[i][j]) * self.fs[j](x) *
                                (logistic(en(x - self.centers[i]), self.alpha) -
                                 logistic(en(x - self.centers[i]), self.alpha) ** 2) for j in range(self.n)])
        funcs = [lambda x: 0] + self.fs + [partial(sean, i) for i in range(self.m)]
        # Least squares preliminaries
        #approx_points = self.centers + [self.centers[i] + np.array([0.1, 0]) for i in range(self.m)] \
        #                + [self.centers[i] + np.array([0, 0.1]) for i in range(self.m)] + \
        #                [self.centers[i] - np.array([0.1, 0]) for i in range(self.m)]\
        #                + [self.centers[i] - np.array([0, 0.1]) for i in range(self.m)]
        approx_points = [np.array([i, j]) for i in np.linspace(0, 1.35, 25) for j in np.linspace(0, 3.49, 25)]
        A = np.array([[logistic(en(approx_points[col] - self.centers[row]), self.alpha) for row in range(self.m)]
                      for col in range(len(approx_points))])
        A = np.hstack([np.array([[1] for i in range(len(approx_points))]),
                       np.array([approx_points[row] for row in range(len(approx_points))]), A])
        # Build our koopman operator
        K = np.zeros((self.npm+1, self.npm+1))
        for row in range(self.npm+1):
            f = np.array([funcs[row](approx_points[i]) for i in range(len(approx_points))])
            K[row] = np.linalg.lstsq(A, f)[0]
        self.koop = K

    def lift_state(self):
        """
        Sets the starting values for the state in the koopman domain.
        :return: 
        """
        init_state = self.x0
        ex0 = np.array(self.x0)
        for mu in self.centers:
            init_state.append(logistic(np.linalg.norm(ex0 - mu)**2, self.alpha))
        return np.hstack([np.array([1]), init_state])

    def simulate_koop(self, times):
        """
        runs a simulation for the koopman operator state
        :param times: array of times to show the state behavior at.
        :return: an array of the behavior of the system...
        """
        ex0 = self.lift_state()
        self.ksim = integrate.odeint(lambda x, t: self.koop.dot(x), ex0, times)

    def sim_koop(self, times):
        ex0 = self.lift_state()
        state = []
        Kd, V = np.linalg.eig(self.koop)
        Vinv = np.linalg.inv(V)

        for time in times:
            ekd = np.diag([np.exp(d*time) for d in Kd])
            state.append(V.dot(ekd).dot(Vinv).dot(ex0))
        self.ksim = np.array(state)
        print(self.ksim.shape)

    def simulate_reg(self, times):
        """
        runs a simulation of the normal system to use for comparison.
        :param times: 
        :return: 
        """
        print(self.fs)
        func = lambda x, t: np.array([f(x) for f in self.fs])
        self.nsim = integrate.odeint(func, self.x0, times)

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


def ef1(x):
    return 1. - x[1]


def ef2(x):
    return x[0]**2 - x[1]**2


def oss1(x):
    return 1. - np.sin(x[0] - x[1])


def oss2(x):
    return 2. - 2. * np.sin(x[1] - x[0])


def oss1(x):
    return x[1]


def oss2(x):
    return -x[0] + 0.2 * (1 - x[0]**2) * x[1]


def __main__():
    fs = [oss1, oss2]
    # x_set = np.hstack([np.linspace(-2.4, 3.01, 7), np.linspace(.9, 1.9, 6)])
    # my_set = np.hstack([np.linspace(-12, 2, 7), np.linspace(-.35, 1.7, 6)])
    centers = [np.array([i, j]) for i in np.linspace(-2, 2, 5) for j in np.linspace(-2, 2, 5)]
    alpha = 10
    starting_vals = [.95, 0.65]
    t = np.linspace(0, 50, 200)

    example = NonlinearSystem(fs, centers, alpha, starting_vals)
    example.learn_koop_learn_pts(100)
    example.simulate_reg(t)
    example.sim_koop(t)
    example.plot(indx=0, koopman=True)
    example.plot(indx=1, koopman=True)
    print(example.ksim[:, 1:3] - example.nsim)
    print("centers", example.centers)
    # Lets see how good our approximations are...
    funcs = [lambda x: 0] + example.fs + [lambda x: sum([2 * (x[j] - example.centers[i][j]) * example.fs[j](x) *
                                                         (logistic(en(x - example.centers[i]), example.alpha) -
                                                          logistic(en(x - example.centers[i]), example.alpha) ** 2) for
                                                         j in range(example.n)]
                                                        ) for i in range(example.m)]

    approx_points = [np.array([i, j]) for i in np.linspace(-2, 2, 20) for j in np.linspace(-2, 2, 20)]
    # approx_points = example.centers + [example.centers[i] + 0.01 for i in range(example.m)] \
    # + [example.centers[i] - 0.01 for i in range(example.m)]
    # approx_points = example.centers + [example.centers[i] + np.array([0.1, 0]) for i in range(example.m)] \
    #                        + [example.centers[i] + np.array([0, 0.1]) for i in range(example.m)] + \
    #                        [example.centers[i] - np.array([0.1, 0]) for i in range(example.m)]\
    #                        + [example.centers[i] - np.array([0, 0.1]) for i in range(example.m)]
    num = 0
    numnum = 0
    for row in example.koop:
        en = lambda y: np.linalg.norm(y) ** 2
        func = lambda x: row[0] + sum([row[i + 1] * x[i] for i in range(example.n)]) + \
                         sum([row[i + example.n + 1] * logistic(en(x - example.centers[i]), example.alpha)
                              for i in range(len(example.centers))])
        if numnum >= 0:
            plt.plot([abs(funcs[num](i) - func(i)) for i in approx_points], label="error")
            plt.plot([funcs[num](i) for i in approx_points], label="exact")
            plt.plot([func(i) for i in approx_points], label="approx")
            plt.title("{0}".format(str(num)))
            plt.legend()
            plt.show()
        num += 1
        numnum += 1
"""to get output: python xxxx.py >>output.txt"""
__main__()



""""""
