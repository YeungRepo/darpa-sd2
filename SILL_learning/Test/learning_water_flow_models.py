import numpy as np
from scipy import linalg as la

A12 = np.array([[-1, 1, 0],
                [0, -1, 1],
                [1, 0, -1]])  # Unknown head nodes incidence matrix
A10 = 3



# Extended Dynamic Mode Decomposition
# Testing the concept
data = np.array([[1, 2],
                 [2, 5],
                 [3, 10],
                 [4, 17],
                 [5, 26],
                 [6, 37]])

sai_1 = lambda x: 1
sai_2 = lambda x: x
sai_3 = lambda x: x**2 - 1
sai_4 = lambda x: x**3 - 3 * x
sai_5 = lambda x: x**4 - 6 * x**2 + 3
sai_6 = lambda x: 1
sai_7 = lambda x: 1
sai_8 = lambda x: 1
sai_9 = lambda x: 1
sai_10 = lambda x: 1
sai_11 = lambda x: 1

sai = lambda x: np.array([sai_1(x), sai_2(x), sai_3(x), sai_4(x), sai_5(x)])

a = [1, 1, 1, 1, 1]
phi = lambda a: sum([sai[i] * a[i] for i in range(5)])

G = 0.2 * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 0]))) for i in range(5)], axis=0)
A = 0.2 * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 1]))) for i in range(5)], axis=0)
print(G.dot(G))
K = np.linalg.inv(G.T.dot(G)).dot(G).dot(A)

# So, K is the HybridKoopman Operator... is that good?  We're happy?
# K is the finite dimensional approximation to the HybridKoopman Operator it maps phi to phi_hat
# So K * phi(x) roughly approximates sai(F(x)) * [1, 1, 1, 1, 1]

# Take a linear time invariant system and bring it to koopman land and then apply the koopman operator to it a few times
# and see how the state evolves, then take the results and bring them back into the normal state space and compare it
# to simulating the original system.

# A,B,C
A = np.array([[-1, 4],
              [0, -2]])
B = np.array([[1, 0],
              [1, 1]])
x0 = np.array([0, 0])
u = np.array([[1, 1],
              [1, 1],
              [1, 1],
              [1, 1]])
# We use the attempt called: HybridKoopman with inputs and control, KIC
# https://arxiv.org/pdf/1602.07647.pdf MAy 30, 2017

C = np.eye(2, 2)
G = np.hstack((A, B))
bot = np.hstack((np.zeros([2, 2]), np.eye(2, 2)))
G = np.vstack((G, bot))
# print(G)
eig_G, evec_G = np.linalg.eig(G)

# HybridKoopman Operator for the simple Hybrid System of a ball bouncing
d = 0.1
K1 = np.array([[0, 1, 0, 0],
               [0, 0, d, -1],
               [0, 0, 2*d, -2],
               [0, 0, 0, 0]])

evals1, evecs1 = la.eig(K1, left=True, right=False)

d = 7

K2 = np.array([[0, 1, 0, 0],
               [0, 0, -d, -1],
               [0, 0, -2*d, -2],
               [0, 0, 0, 0]])

evals2, evecs2 = la.eig(K2, left=True, right=False)

from sympy import Matrix


A = Matrix([[0, 1, 0, 0],
            [0, 0, 7, -1],
            [0, 0, 14, -2],
            [0, 0, 0, 0]])

print(A.T.nullspace())


# Before for the ball bouncing system I had an incorrect HybridKoopman operator
# I think this is a truncated version of the correct one:
g = -9.8
d = 0.2
K1 = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, g, 0, d, 0, 0, 0, 0],
               [0, 0, 2*g, 0, 2*d, 0, 0, 0],
               [0, 0, 0, 3*g, 0, 3*d, 0, 0],
               [0, 0, 0, 0, 4*g, 0, 4*d, 0],
               [0, 0, 0, 0, 0, 5*g, 0, 5*d],
               [0, 0, 0, 0, 0, 0, 6*g, 0]])
print(K1)

evals1, evecs1 = la.eig(K1, left=True, right=False)

g = -9.8
d2 = -0.2
K2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, g, 0, d2, 0, 0, 0, 0],
               [0, 0, 2*g, 0, 2*d2, 0, 0, 0],
               [0, 0, 0, 3*g, 0, 3*d2, 0, 0],
               [0, 0, 0, 0, 4*g, 0, 4*d2, 0],
               [0, 0, 0, 0, 0, 5*g, 0, 5*d2],
               [0, 0, 0, 0, 0, 0, 6*g, 0]])
print(K2)

evals2, evecs2 = la.eig(K1, left=True, right=False)


# Let's try something completely different, June 1st 2017
A = np.array([[1, -1, 0, 0],
              [1, 0, 0, -1],
              [0, 0, 1, -1],
              [0, -1, 1, 0]])
A = np.array([[0, -1, 0, 0],
              [1, 0, 0, -1],
              [0, 0, 0, -1],
              [0, -1, 1, 0]])
x0 = np.array([1, 0, 0, 0])

x = x0
state = []
for i in range(25):
    state.append(x)
    x = A.dot(x)

print(state)
# This is a six cyclic thing

# Here is a simple 2 dynamic system:
# D1: x[t] = 2x[t-1]
# D2: x[t] = -x[t-1]

A = np.array([[0, 2, 0, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 2],
              [.5, 0, 0, 0]])

A = np.array([[0, 0, 0, .5],
              [2, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 2, 0]])

x = x0
state = []
for i in range(25):
    state.append(x)
    x = A.dot(x)

print(state)

A = np.array([[-.5, 0, 0, 0, 0],
              [0, -.51, 0, 0, 0],
              [1, -1, 0, 0, 0],
              [0, 0, 1, -.3, 0],
              [0, 0, 1, 0, -.6]])

x0 = np.array([1, 1, 0, 0, 0])
x = x0
state = []
for i in range(25):
    state.append(x)
    x = A.dot(x)

print(state)
















