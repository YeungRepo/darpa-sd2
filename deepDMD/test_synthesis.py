#! /usr/bin/python

### Import Packages 
import pickle; # for data I/O

# Math Packages 
import numpy as np;
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
import math;
import random;
from numpy import genfromtxt


# Import CVXOPT Packages
from cvxpy import Minimize, Problem, Variable,norm1,norm2,installed_solvers,lambda_max;
from cvxpy import norm as cvxpynorm;
import cvxpy;

# Tensorflow Packages
import tensorflow as tf

def test_synthesis(A,B):

    # Problem data.

    N_u = B.shape[1]
    N_x = A.shape[0]
    Cv = np.eye(N_u)
    Av = cvxpy.Variable(rows=N_u,cols=N_u);
    Bv = cvxpy.Variable(rows=N_u,cols=N_x);

    P1 = np.eye(A.shape[0]);
    P2 = np.eye(Cv.shape[0]);
    
    
    # Construct the problem.

    objective = cvxpy.Minimize(cvxpy.norm1(Av) + cvxpy.norm1(Bv))

    SystemMatrix1 = cvxpy.hstack([P1, np.zeros((N_x,N_u)), A.T, Bv.T]);
    SystemMatrix2 = cvxpy.hstack([np.zeros( ( Cv.shape[0],N_x ) ),P2,np.matmul(Cv.T,B.T),Av.T]);
    SystemMatrix3 = cvxpy.hstack([A, np.matmul(B,Cv), P1, np.zeros( (A.shape[0],P2.shape[0]) )]);
    SystemMatrix4 = cvxpy.hstack([Bv, Av, np.zeros((N_u,N_x) )  ,P2]); 
    SystemMatrix = cvxpy.vstack( [SystemMatrix1,SystemMatrix2,SystemMatrix3,SystemMatrix4]);
    constraints = [0<SystemMatrix]; 

    prob = cvxpy.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver='ECOS',verbose=True)
    return result, Av.value, Bv.value
    # The optimal value for x is stored in x.value.
    #  The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    #print constraints[0].dual_value


