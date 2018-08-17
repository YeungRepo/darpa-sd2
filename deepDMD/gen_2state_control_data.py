


from sklearn.linear_model import Lasso, LassoLarsCV, LassoLars, LassoCV

import cvxopt
from cvxpy import Minimize,Problem,Variable,norm1,norm2,installed_solvers,lambda_max
from cvxpy import norm as cvxpynorm

from numpy.linalg import pinv,inv,cond,svd,eig

from numpy import cov,matrix,kron
from scipy.linalg import norm
from scipy import random

from numpy.polynomial.hermite import hermvander
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def zhang_system(tsteps=500,init=[0.5,-0.1]):
    a1 = -0.96;
    a2 = 0.88;
    a3 = -0.95;
    freq = -2.0;
    def f(xp,up):
        xf1 = -a1*xp[1] + a3*xp[0]*np.sin(freq*up);
        xf2 = np.sin(freq*xp[0]) + a2*xp[1] + xp[1]*xp[0]*up;
        #xf1 = -a1*xp[1] + 0.0*up;
        #xf2 = np.sin(-freq*xp[0] - xp[1]) + a2*xp[1] -xp[1]*up;
        return [xf1,xf2];
    X = [];
    
    X.append(init);
    xp_val = init;
    
    u_seq = [0.2]*(tsteps/4)+[0.0]*(tsteps*3/4);

    for i in range(0,tsteps-1):
        xf_val = f(xp_val,u_seq[i]);
        X.append(xf_val);
        xp_val = xf_val;
    
    X = np.asarray(X);


    tspan = np.arange(0,tsteps);
    X = np.transpose(X);
    print "[DEBUG] :" + repr( X.shape);
    fig,ax = plt.subplots(nrows=1,ncols=1);
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Magnitude')
    for i in range(2):
        alabel = 'x'+str(i)
        ax.plot(tspan,X[i,:],'-*',label=alabel);

    ax.legend(loc='upper right', shadow=True, fontsize='smaller')
    plt.show()

    return X,u_seq;
    




obs_channels = 2
full_steps = 500;
X,u_seq = zhang_system(tsteps=full_steps,init=[0.3,-0.1]);

print X.shape


Yf = X[0:obs_channels,1:];
Yp = X[0:obs_channels,0:-1];

import pickle

file_obj = file('koopman_data/zhang_control.pickle','wb');


pickle.dump([list(np.transpose(Yp)),list(np.transpose(Yf)),list(u_seq)],file_obj);
file_obj.close();


    

