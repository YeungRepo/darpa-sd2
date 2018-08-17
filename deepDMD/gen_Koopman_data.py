#! /usr/env/bin python

import pickle;
import numpy as np;

#to be replaced with cmd line args 
hidden_vars = 2;
max_dim = 4;
# to be replaced with cmd line args 

min_dim = max_dim;
max_t = 120.0;
min_t = 0.0;
num_points =1000;
n_dims = max_dim;#np.random.randint(min_dim,max_dim)*2;
noise_factor = 0.0;

def rand_A_osc(n_dims,noise):
    yinit = list(np.random.rand(n_dims));
    #create random full matrix with complex conjugate pair column vectors
    T_seed = .1*np.random.rand(n_dims,n_dims/2) + .01*1j*np.random.rand(n_dims,n_dims/2)
    #print "[DEBUG] T_seed.shape:" + repr(T_seed.shape);
    T = np.append(T_seed,np.conj(T_seed),axis=1)
    
    #create random diagonal matrix of complex eigenvalues with negative real parts in complex conjugate pairs
    real_eig_seed = -np.random.randint(0,1,size=n_dims/2)*abs(np.random.rand(n_dims/2));
    #print "[DEBUG] real_eig_seed: " +repr(real_eig_seed);
    imag_eig_seed = 1j*np.random.randint(0,n_dims,size=n_dims/2)*abs(np.random.rand(n_dims/2));
    eig_seed = real_eig_seed+ imag_eig_seed;
    Dn_diag = np.append(eig_seed,np.conj(eig_seed))
    Dn = np.diag(Dn_diag)/np.random.randint(1,n_dims/2)

    #print "[DEBUG] eig(Dn): " + repr(np.linalg.eig(Dn));

    
    #create interaction matrix TDT^-1
    #print "[DEBUG] T.shape: " + repr( T.shape);
    #print "[DEBUG] Dn.shape:" + repr(Dn.shape);
    Qclean = np.dot(T,np.dot(Dn,np.linalg.inv(T))) ;
    noise_term = noise*(-1.0)*abs(np.random.rand(n_dims,n_dims));
    Q = Qclean+noise_term;
    #print "[DEBUG] Q: " +repr(np.real(Q));
    return np.real(Q),yinit


A,yinit = rand_A_osc(n_dims,noise_factor);

#print "[DEBUG] np.linalg.eig(A): " +repr(np.linalg.eig(A));

def prop_discrete_linear_ds(x0,t,A):
    dxdt = np.dot(A,x0)
    return dxdt;

from scipy.integrate import odeint
import matplotlib.pyplot as plt

x0 = np.random.rand(n_dims);
print "[INFO] state dimension: " + repr(n_dims);
t = np.linspace(min_t,max_t,num_points);
sol = odeint(prop_discrete_linear_ds,x0,t,args=(A,));
plt.plot(t,sol[:,0],'r.-',label='x_1(t)',markersize=10);
plt.plot(t,sol[:,1],'b.-',label='x_2(t)',markersize=10);
plt.plot(t,sol[:,2],'g.-',label='x_3(t)',markersize=10);
plt.legend(loc='best');
plt.xlabel('t');
import matplotlib
matplotlib.rcParams.update({'font.size':22})

plt.grid();
plt.show();


#print "[DEBUG] sol.shape: " + repr(sol.shape);
#print "[DEBUG] len(list(sol)): " +repr(len(list(sol)));


# # # - - - save vars to binary file - - - # # #
sol = sol[:,0:n_dims-hidden_vars]/np.max(sol);


    

Ywhole = list(sol);



Yp = Ywhole[0:num_points-1];
Yf = Ywhole[1:num_points];

import pickle;
file_path = 'koopman_data/rand_osc.pickle'

file_obj = file(file_path,'wb');
pickle.dump([Yp,Yf],file_obj);


