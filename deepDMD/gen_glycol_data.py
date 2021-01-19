











from sklearn.linear_model import Lasso, LassoLarsCV, LassoLars, LassoCV

#import cvxopt
#from cvxpy import Minimize,Problem,Variable,norm1,norm2,installed_solvers,lambda_max
#from cvxpy import norm as cvxpynorm

from numpy.linalg import pinv,inv,cond,svd,eig

from numpy import cov,matrix,kron
from scipy.linalg import norm
from scipy import random

from numpy.polynomial.hermite import hermvander
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def glycol(tsteps=101,time=5,init=[.15,.19,.04,.1,.08,.14,.05],train_steps=50,plot_channels=3):
     
    J = 2.5
    A = 4
    N = 1
    K1 = .52
    kap = 13
    phi = .1
    q = 4
    k = 1.8
    k1 = 100
    k2 = 6
    k3 = 16
    k4 = 100
    k5 = 1.28
    k6 = 12
 
    
    def f(y,t):
        dy0 = J - (k1*y[0]*y[5]/(1+(y[5]/K1)**q))
        dy1 = 2*(k1*y[0]*y[5]/(1+(y[5]/K1)**q)) - k2*y[1]*(N-y[4]) - k6*y[1]*y[4]
        dy2 = k2*y[1]*(N-y[4]) - k3*y[2]*(A-y[5])
        dy3 = k3*y[2]*(A-y[5]) - k4*y[3]*y[4]-kap*(y[3]-y[6])
        dy4 = k2*y[1]*(N-y[4]) - k4*y[3]*y[4] - k6*y[1]*y[4]
        dy5 = -2*(k1*y[0]*y[5]/(1+(y[5]/K1)**q)) + 2*k3*y[2]*(A-y[5]) - k5*y[5]
        dy6 = phi*kap*(y[3]-y[6]) - k*y[6]

        return [dy0,dy1,dy2,dy3,dy4,dy5,dy6]

    tspan = np.linspace(0,time,tsteps)
    Y = odeint(f,init,tspan)
    Y=np.transpose(Y)
    
    #Truncated for later prediction
    Yt = Y[:,0:Y.shape[1]-(tsteps-train_steps)]
    
    fig,ax = plt.subplots(nrows=1,ncols=1);
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Magnitude')
    for i in range(plot_channels):
        alabel = 'x'+str(i)
        ax.plot(tspan,Y[i,:],'.',label=alabel);

    ax.legend(loc='upper right', shadow=True, fontsize='smaller')


    plt.show()
    
    return (Y,Yt)



obs_channels = 7
plot_channels =4
full_steps=900;
train_steps = 200
est_steps = 200
time = 5.0
    
#init=[.15-1.6,.19-2.16,.04-.2,.1-.35,.08-.3,.14-2.67,.05-.1]
X,Xt = glycol(tsteps=full_steps,time=time,init=[1,.19,.2,.1,.3,.14,.05],train_steps=train_steps,plot_channels=plot_channels)


print(X.shape)
#Yf = X[:,1:];
#Yp = X[:,0:-1];

obs_channels = 7;
Yf = X[0:obs_channels,1:];
Yp = X[0:obs_channels,0:-1];

import pickle

file_obj = open('koopman_data/glycol.pickle','wb');

print(Yp.shape)
pickle.dump([list(np.transpose(Yp)),list(np.transpose(Yf))],file_obj);
file_obj.close();


    

